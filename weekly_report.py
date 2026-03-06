"""
weekly_report.py
直近1週間の chat_history_log.csv を集計・分析し、
PDFレポートを生成して Slack に送信する。

実行方法:
  python3 weekly_report.py                   # 直近7日分
  python3 weekly_report.py --days 14         # 直近14日分
  python3 weekly_report.py --dry-run         # Slack送信せずPDFのみ生成
"""

import argparse
import csv
import io
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import anthropic
import httpx
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------- パス設定 ----------

BASE_DIR         = Path(__file__).parent
LOGS_DIR         = BASE_DIR / "logs"
CHAT_HISTORY_CSV = LOGS_DIR / "chat_history_log.csv"
REPORT_DIR       = LOGS_DIR / "reports"

# ---------- .env 読み込み ----------

def _force_load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ[k.strip()] = v.strip()

_force_load_env(BASE_DIR / ".env")

# ---------- フォント設定（日本語対応） ----------

def _register_japanese_font() -> str:
    """システムの日本語フォントを登録して名前を返す。見つからなければ Helvetica を返す。"""
    candidates = [
        # macOS
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode MS.ttf",
        # Linux (GitHub Actions / Streamlit Cloud)
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont("JpFont", path))
                return "JpFont"
            except Exception:
                continue
    return "Helvetica"


# ---------- CSV 読み込み ----------

def load_csv(days: int) -> list[dict]:
    """直近 days 日分のログを返す。"""
    if not CHAT_HISTORY_CSV.exists():
        return []
    cutoff = datetime.now() - timedelta(days=days)
    rows = []
    with open(CHAT_HISTORY_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = datetime.fromisoformat(row["timestamp"])
                if ts >= cutoff:
                    rows.append(row)
            except Exception:
                continue
    return rows


# ---------- 集計 ----------

def aggregate(rows: list[dict]) -> dict:
    total      = len(rows)
    resolved   = sum(1 for r in rows if r.get("feedback") == "resolved")
    unresolved = sum(1 for r in rows if r.get("feedback") == "unresolved")
    no_fb      = total - resolved - unresolved
    resolution_rate = round(resolved / total * 100, 1) if total > 0 else 0

    # 日別件数
    daily: dict[str, int] = {}
    for r in rows:
        try:
            day = r["timestamp"][:10]
            daily[day] = daily.get(day, 0) + 1
        except Exception:
            pass

    # 未解決 / フィードバックなしの質問一覧
    problem_rows = [r for r in rows if r.get("feedback") in ("unresolved", "", None)]

    return {
        "total": total,
        "resolved": resolved,
        "unresolved": unresolved,
        "no_feedback": no_fb,
        "resolution_rate": resolution_rate,
        "daily": dict(sorted(daily.items())),
        "problem_rows": problem_rows,
    }


# ---------- AI 分析 ----------

def ai_analysis(rows: list[dict], stats: dict) -> str:
    """Claude Sonnet で問題のある質問を分析し、改善アドバイスを返す。"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "（ANTHROPIC_API_KEY が未設定のため分析スキップ）"

    problem_rows = stats["problem_rows"]
    if not problem_rows:
        return "未解決・フィードバックなしの質問はありませんでした。引き続き品質を維持してください。"

    # 分析対象は最大15件（トークン節約）
    sample = problem_rows[:15]
    cases = "\n".join(
        f"[{i+1}] Q: {r['question']}\n    A（抜粋）: {r['answer'][:120]}...\n    フィードバック: {r.get('feedback') or 'なし'}"
        for i, r in enumerate(sample)
    )

    prompt = f"""あなたは通販コールセンターAIシステムの品質改善コンサルタントです。
以下は直近1週間のコールセンターAIの「未解決またはフィードバックなし」の質問・回答ペアです。

{cases}

上記を分析して、以下の形式で日本語でレポートしてください：

## 1. 問題パターンの分類
質問を2〜4つのパターンに分類し、各パターンの件数と傾向を説明する。

## 2. 回答精度が低い原因
なぜ正確な回答ができなかったのか、具体的な原因を3点以内で挙げる。

## 3. 改善アドバイス
知識ベースやプロンプトをどう修正すれば回答精度が上がるか、具体的なアクションを3点提案する。

## 4. 優先対応が必要な質問トップ3
最も早急に対処すべき質問を3つ挙げ、それぞれの改善案を1文で述べる。

簡潔・箇条書き中心で、800文字以内にまとめてください。"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    except Exception as e:
        return f"（AI分析エラー: {e}）"


# ---------- PDF 生成 ----------

def build_pdf(stats: dict, analysis_text: str, period_label: str, days: int) -> bytes:
    """レポートPDFをバイト列で返す。"""
    font_name = _register_japanese_font()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    # スタイル定義
    base = getSampleStyleSheet()
    style_title = ParagraphStyle(
        "title", fontName=font_name, fontSize=18, leading=24,
        alignment=TA_CENTER, textColor=colors.HexColor("#1A237E"),
        spaceAfter=4 * mm,
    )
    style_subtitle = ParagraphStyle(
        "subtitle", fontName=font_name, fontSize=10,
        alignment=TA_CENTER, textColor=colors.grey, spaceAfter=6 * mm,
    )
    style_h2 = ParagraphStyle(
        "h2", fontName=font_name, fontSize=13, leading=18,
        textColor=colors.HexColor("#1565C0"), spaceBefore=6 * mm, spaceAfter=3 * mm,
    )
    style_body = ParagraphStyle(
        "body", fontName=font_name, fontSize=9, leading=14,
        alignment=TA_LEFT, spaceAfter=2 * mm,
    )
    style_small = ParagraphStyle(
        "small", fontName=font_name, fontSize=8, leading=12,
        textColor=colors.HexColor("#555555"),
    )

    story = []

    # ── タイトル ──
    story.append(Paragraph("📊 週次品質レポート", style_title))
    story.append(Paragraph(f"集計期間：{period_label}（直近{days}日間）", style_subtitle))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#BBDEFB")))
    story.append(Spacer(1, 4 * mm))

    # ── サマリー表 ──
    story.append(Paragraph("1. 集計サマリー", style_h2))
    summary_data = [
        ["指標", "件数 / 率"],
        ["総質問数", str(stats["total"])],
        ["✅ 解決済み", f"{stats['resolved']}件"],
        ["❌ 未解決", f"{stats['unresolved']}件"],
        ["フィードバックなし", f"{stats['no_feedback']}件"],
        ["解決率", f"{stats['resolution_rate']}%"],
    ]
    tbl = Table(summary_data, colWidths=[80 * mm, 60 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#1565C0")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, -1), font_name),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#BBDEFB")),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 4 * mm))

    # ── 日別質問数 ──
    story.append(Paragraph("2. 日別質問数", style_h2))
    if stats["daily"]:
        daily_data = [["日付", "件数"]] + [[k, str(v)] for k, v in stats["daily"].items()]
        dtbl = Table(daily_data, colWidths=[50 * mm, 30 * mm])
        dtbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#42A5F5")),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, -1), font_name),
            ("FONTSIZE",     (0, 0), (-1, -1), 9),
            ("ALIGN",        (1, 0), (1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
            ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#BBDEFB")),
            ("TOPPADDING",   (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ]))
        story.append(dtbl)
    else:
        story.append(Paragraph("（該当データなし）", style_body))
    story.append(Spacer(1, 4 * mm))

    # ── 未解決質問一覧 ──
    story.append(Paragraph("3. 未解決・要確認の質問一覧", style_h2))
    problem_rows = stats["problem_rows"]
    if problem_rows:
        prob_data = [["#", "日時", "質問（抜粋）", "フィードバック"]]
        for i, r in enumerate(problem_rows[:20], 1):
            q_short = r["question"][:30] + ("…" if len(r["question"]) > 30 else "")
            fb = r.get("feedback") or "なし"
            prob_data.append([str(i), r["timestamp"][:16], q_short, fb])
        ptbl = Table(prob_data, colWidths=[8 * mm, 32 * mm, 95 * mm, 18 * mm])
        ptbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#EF5350")),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, -1), font_name),
            ("FONTSIZE",     (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FFF5F5")]),
            ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#FFCDD2")),
            ("TOPPADDING",   (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
            ("WORDWRAP",     (2, 1), (2, -1), True),
        ]))
        story.append(ptbl)
    else:
        story.append(Paragraph("✅ 未解決の質問はありませんでした。", style_body))
    story.append(Spacer(1, 4 * mm))

    # ── AI 分析 ──
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#BBDEFB")))
    story.append(Paragraph("4. AI改善アドバイス（Claude分析）", style_h2))
    # Markdown の ## を除去してParagraphに渡す
    clean_analysis = analysis_text.replace("##", "").replace("**", "")
    for line in clean_analysis.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 2 * mm))
        else:
            story.append(Paragraph(line, style_body))

    story.append(Spacer(1, 6 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}　｜　Q&Ai コールセンター品質管理レポート",
        style_small,
    ))

    doc.build(story)
    return buf.getvalue()


# ---------- Slack 送信 ----------

def send_to_slack(pdf_bytes: bytes, filename: str, stats: dict, period_label: str) -> bool:
    """Slack Incoming Webhook + files.getUploadURLExternal でPDFを送信する。"""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    slack_channel = os.getenv("SLACK_CHANNEL", "#general")

    if not webhook_url:
        print("[WARN] SLACK_WEBHOOK_URL が未設定です。Slack送信をスキップします。")
        return False

    # ── テキスト通知（Webhook）──
    rate_emoji = "✅" if stats["resolution_rate"] >= 80 else "⚠️" if stats["resolution_rate"] >= 60 else "🚨"
    text = (
        f"*📊 週次品質レポート*　{period_label}\n"
        f"```\n"
        f"総質問数       : {stats['total']}件\n"
        f"解決済み       : {stats['resolved']}件\n"
        f"未解決         : {stats['unresolved']}件\n"
        f"解決率         : {rate_emoji} {stats['resolution_rate']}%\n"
        f"```\n"
        f"詳細レポート（PDF）を添付しています。改善アドバイスをご確認ください。"
    )
    try:
        resp = httpx.post(webhook_url, json={"text": text}, timeout=10)
        resp.raise_for_status()
        print(f"[OK] Slack テキスト通知送信完了")
    except Exception as e:
        print(f"[ERROR] Slack Webhook 送信失敗: {e}")
        return False

    # ── PDF ファイル添付（Bot Token がある場合のみ）──
    if slack_token:
        try:
            # Step1: アップロードURL取得
            r1 = httpx.post(
                "https://slack.com/api/files.getUploadURLExternal",
                headers={"Authorization": f"Bearer {slack_token}"},
                data={"filename": filename, "length": len(pdf_bytes)},
                timeout=15,
            )
            r1.raise_for_status()
            j1 = r1.json()
            if not j1.get("ok"):
                print(f"[ERROR] files.getUploadURLExternal: {j1.get('error')}")
                return True  # テキスト通知は成功しているので True

            upload_url = j1["upload_url"]
            file_id    = j1["file_id"]

            # Step2: ファイルアップロード
            r2 = httpx.post(upload_url, content=pdf_bytes, timeout=30)
            r2.raise_for_status()

            # Step3: チャンネルに公開
            r3 = httpx.post(
                "https://slack.com/api/files.completeUploadExternal",
                headers={"Authorization": f"Bearer {slack_token}"},
                json={
                    "files": [{"id": file_id, "title": filename}],
                    "channel_id": slack_channel,
                },
                timeout=15,
            )
            r3.raise_for_status()
            j3 = r3.json()
            if j3.get("ok"):
                print(f"[OK] PDF ファイル送信完了: {filename}")
            else:
                print(f"[WARN] PDF送信部分失敗: {j3.get('error')}")
        except Exception as e:
            print(f"[WARN] PDF添付失敗（テキスト通知は送信済み）: {e}")
    else:
        print("[INFO] SLACK_BOT_TOKEN 未設定のためPDF添付スキップ（テキスト通知のみ）")

    return True


# ---------- メイン ----------

def run(days: int = 7, dry_run: bool = False) -> tuple[bool, str]:
    """
    レポートを生成して Slack 送信する。
    戻り値: (成功フラグ, メッセージ)
    """
    print(f"[START] 週次レポート生成 (直近{days}日)")

    # 1. データ読み込み
    rows = load_csv(days)
    if not rows:
        msg = f"直近{days}日のログがありません。レポート生成をスキップします。"
        print(f"[INFO] {msg}")
        return False, msg

    # 2. 集計
    stats = aggregate(rows)
    print(f"[INFO] 集計完了: {stats['total']}件 / 解決率 {stats['resolution_rate']}%")

    # 3. AI 分析
    print("[INFO] AI分析中...")
    analysis = ai_analysis(rows, stats)

    # 4. PDF 生成
    now = datetime.now()
    period_label = f"{(now - timedelta(days=days)).strftime('%Y/%m/%d')} 〜 {now.strftime('%Y/%m/%d')}"
    pdf_bytes = build_pdf(stats, analysis, period_label, days)
    filename = f"qai_weekly_report_{now.strftime('%Y%m%d')}.pdf"

    # 5. PDF 保存（ローカル）
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = REPORT_DIR / filename
    pdf_path.write_bytes(pdf_bytes)
    print(f"[INFO] PDF保存: {pdf_path}")

    # 6. Slack 送信
    if dry_run:
        print("[DRY-RUN] Slack送信をスキップしました。PDFはローカルに保存されています。")
        return True, f"PDF生成完了（dry-run）: {pdf_path}"

    ok = send_to_slack(pdf_bytes, filename, stats, period_label)
    if ok:
        return True, f"レポート送信完了: {filename}"
    else:
        return False, "Slack送信失敗（PDFはローカルに保存済み）"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="週次品質レポートを生成してSlackに送信")
    parser.add_argument("--days",    type=int, default=7,  help="集計対象の日数（デフォルト: 7）")
    parser.add_argument("--dry-run", action="store_true",  help="Slack送信せずPDFのみ生成")
    args = parser.parse_args()

    success, message = run(days=args.days, dry_run=args.dry_run)
    print(f"[{'OK' if success else 'FAIL'}] {message}")
    sys.exit(0 if success else 1)
