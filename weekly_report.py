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
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import anthropic
import httpx
from fpdf import FPDF, XPos, YPos

# ---------- パス設定 ----------

BASE_DIR         = Path(__file__).parent
LOGS_DIR         = BASE_DIR / "logs"
CHAT_HISTORY_CSV = LOGS_DIR / "chat_history_log.csv"
REPORT_DIR       = LOGS_DIR / "reports"
FONT_DIR         = BASE_DIR / "data" / "fonts"

# フォントファイルのパス候補（優先順）
FONT_CANDIDATES = [
    FONT_DIR / "NotoSansCJKjp-Regular.otf",
    FONT_DIR / "NotoSansCJKjp-Regular.ttf",
    # Linux (GitHub Actions)
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf"),
]

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

    daily: dict[str, int] = {}
    for r in rows:
        try:
            day = r["timestamp"][:10]
            daily[day] = daily.get(day, 0) + 1
        except Exception:
            pass

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
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "ANTHROPIC_API_KEY が未設定のため分析スキップ"

    problem_rows = stats["problem_rows"]
    if not problem_rows:
        return "未解決・フィードバックなしの質問はありませんでした。引き続き品質を維持してください。"

    sample = problem_rows[:15]
    cases = "\n".join(
        f"[{i+1}] Q: {r['question']}\n    A（抜粋）: {r['answer'][:120]}...\n    フィードバック: {r.get('feedback') or 'なし'}"
        for i, r in enumerate(sample)
    )

    prompt = f"""あなたは通販コールセンターAIシステムの品質改善コンサルタントです。
以下は直近1週間のコールセンターAIの「未解決またはフィードバックなし」の質問・回答ペアです。

{cases}

上記を分析して、以下の形式で日本語でレポートしてください：

【1. 問題パターンの分類】
質問を2〜4つのパターンに分類し、各パターンの件数と傾向を説明する。

【2. 回答精度が低い原因】
なぜ正確な回答ができなかったのか、具体的な原因を3点以内で挙げる。

【3. 改善アドバイス】
知識ベースやプロンプトをどう修正すれば回答精度が上がるか、具体的なアクションを3点提案する。

【4. 優先対応が必要な質問トップ3】
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
        return f"AI分析エラー: {e}"


# ---------- PDF生成（fpdf2 + 日本語フォント） ----------

class ReportPDF(FPDF):
    def __init__(self, font_path: Path):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.font_path = str(font_path)
        self.add_font("JP", style="", fname=self.font_path)
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(20, 20, 20)

    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("JP", size=7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 5, f"Q&Ai コールセンター品質管理レポート　生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

    def section_title(self, text: str):
        self.ln(4)
        self.set_font("JP", size=13)
        self.set_text_color(21, 101, 192)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(187, 222, 251)
        self.line(self.get_x(), self.get_y(), self.get_x() + 170, self.get_y())
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def body_text(self, text: str, size: int = 9):
        self.set_font("JP", size=size)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 6, text)

    def colored_table(self, headers: list, rows: list, col_widths: list,
                      header_color: tuple = (21, 101, 192)):
        self.set_font("JP", size=8)
        # ヘッダー
        self.set_fill_color(*header_color)
        self.set_text_color(255, 255, 255)
        for i, (h, w) in enumerate(zip(headers, col_widths)):
            self.cell(w, 7, h, border=1, fill=True, align="C")
        self.ln()
        # データ行
        self.set_text_color(30, 30, 30)
        for j, row in enumerate(rows):
            fill = j % 2 == 1
            self.set_fill_color(245, 247, 250) if fill else self.set_fill_color(255, 255, 255)
            for i, (val, w) in enumerate(zip(row, col_widths)):
                self.cell(w, 6, str(val), border=1, fill=fill, align="C" if i != 2 else "L")
            self.ln()
        self.ln(2)


def find_font() -> Optional[Path]:
    for p in FONT_CANDIDATES:
        if p.exists():
            return p
    return None


def build_pdf(stats: dict, analysis_text: str, period_label: str, days: int) -> bytes:
    font_path = find_font()
    if font_path is None:
        raise FileNotFoundError(
            "日本語フォントが見つかりません。"
            f"以下のいずれかに配置してください:\n" +
            "\n".join(str(p) for p in FONT_CANDIDATES)
        )

    pdf = ReportPDF(font_path)
    pdf.add_page()

    # ── タイトル ──
    pdf.set_font("JP", size=18)
    pdf.set_text_color(26, 35, 126)
    pdf.cell(0, 12, "週次品質レポート", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("JP", size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, f"集計期間：{period_label}（直近{days}日間）", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_draw_color(187, 222, 251)
    pdf.line(20, pdf.get_y() + 2, 190, pdf.get_y() + 2)
    pdf.ln(6)

    # ── 1. サマリー ──
    pdf.section_title("1. 集計サマリー")
    summary_rows = [
        ["総質問数", str(stats["total"])],
        ["解決済み", f"{stats['resolved']}件"],
        ["未解決", f"{stats['unresolved']}件"],
        ["フィードバックなし", f"{stats['no_feedback']}件"],
        ["解決率", f"{stats['resolution_rate']}%"],
    ]
    pdf.colored_table(
        headers=["指標", "件数 / 率"],
        rows=[[r[0], r[1]] for r in summary_rows],
        col_widths=[85, 60],
    )

    # ── 2. 日別質問数 ──
    pdf.section_title("2. 日別質問数")
    if stats["daily"]:
        pdf.colored_table(
            headers=["日付", "件数"],
            rows=[[k, str(v)] for k, v in stats["daily"].items()],
            col_widths=[50, 30],
            header_color=(66, 165, 245),
        )
    else:
        pdf.body_text("（該当データなし）")

    # ── 3. 未解決一覧 ──
    pdf.section_title("3. 未解決・要確認の質問一覧")
    problem_rows = stats["problem_rows"]
    if problem_rows:
        table_rows = []
        for i, r in enumerate(problem_rows[:20], 1):
            q = r["question"][:28] + ("…" if len(r["question"]) > 28 else "")
            fb = r.get("feedback") or "なし"
            table_rows.append([str(i), r["timestamp"][:16], q, fb])
        pdf.colored_table(
            headers=["#", "日時", "質問（抜粋）", "FB"],
            rows=table_rows,
            col_widths=[8, 35, 100, 17],
            header_color=(239, 83, 80),
        )
    else:
        pdf.body_text("未解決の質問はありませんでした。")

    # ── 4. AI分析 ──
    pdf.section_title("4. AI改善アドバイス（Claude分析）")
    # 実効幅 = A4幅(210) - 左右マージン(20*2) = 170mm
    effective_w = 170

    import re as _re
    def _strip_md(text: str) -> str:
        """Markdownの **太字** / # 見出し記号を除去する。"""
        text = _re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)  # **bold** → bold
        text = _re.sub(r'^#+\s*', '', text)                      # # heading → heading
        text = text.replace("---", "")
        return text.strip()

    for line in analysis_text.split("\n"):
        line = _strip_md(line)
        if not line:
            pdf.ln(2)
        elif line.startswith("【"):
            pdf.set_font("JP", size=10)
            pdf.set_text_color(21, 101, 192)
            pdf.multi_cell(effective_w, 7, line)
            pdf.set_text_color(30, 30, 30)
        else:
            pdf.set_font("JP", size=9)
            pdf.multi_cell(effective_w, 6, line)

    return bytes(pdf.output())


# ---------- Slack 送信 ----------

def send_to_slack(pdf_bytes: bytes, filename: str, stats: dict, period_label: str) -> bool:
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    slack_channel = os.getenv("SLACK_CHANNEL", "#general")

    if not webhook_url:
        print("[WARN] SLACK_WEBHOOK_URL が未設定です。Slack送信をスキップします。")
        return False

    rate_emoji = "✅" if stats["resolution_rate"] >= 80 else "⚠️" if stats["resolution_rate"] >= 60 else "🚨"
    text = (
        f"*週次品質レポート*　{period_label}\n"
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
        print("[OK] Slack テキスト通知送信完了")
    except Exception as e:
        print(f"[ERROR] Slack Webhook 送信失敗: {e}")
        return False

    if slack_token:
        try:
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
                return True
            r2 = httpx.post(j1["upload_url"], content=pdf_bytes, timeout=30)
            r2.raise_for_status()
            r3 = httpx.post(
                "https://slack.com/api/files.completeUploadExternal",
                headers={"Authorization": f"Bearer {slack_token}"},
                json={"files": [{"id": j1["file_id"], "title": filename}], "channel_id": slack_channel},
                timeout=15,
            )
            r3.raise_for_status()
            if r3.json().get("ok"):
                print(f"[OK] PDF ファイル送信完了: {filename}")
        except Exception as e:
            print(f"[WARN] PDF添付失敗（テキスト通知は送信済み）: {e}")
    else:
        print("[INFO] SLACK_BOT_TOKEN 未設定のためPDF添付スキップ")

    return True


# ---------- メイン ----------

def run(days: int = 7, dry_run: bool = False) -> tuple[bool, str]:
    print(f"[START] 週次レポート生成 (直近{days}日)")

    rows = load_csv(days)
    if not rows:
        msg = f"直近{days}日のログがありません。レポート生成をスキップします。"
        print(f"[INFO] {msg}")
        return False, msg

    stats = aggregate(rows)
    print(f"[INFO] 集計完了: {stats['total']}件 / 解決率 {stats['resolution_rate']}%")

    print("[INFO] AI分析中...")
    analysis = ai_analysis(rows, stats)

    now = datetime.now()
    period_label = f"{(now - timedelta(days=days)).strftime('%Y/%m/%d')} 〜 {now.strftime('%Y/%m/%d')}"

    print("[INFO] PDF生成中...")
    try:
        pdf_bytes = build_pdf(stats, analysis, period_label, days)
    except FileNotFoundError as e:
        return False, str(e)

    filename = f"qai_weekly_report_{now.strftime('%Y%m%d')}.pdf"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = REPORT_DIR / filename
    pdf_path.write_bytes(pdf_bytes)
    print(f"[INFO] PDF保存: {pdf_path}")

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
