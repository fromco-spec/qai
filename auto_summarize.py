"""
auto_summarize.py
Notionの3DB（施策・マニュアル・料金表）を巡回し、
  ① サマリーが空欄のページ
  ② 最終編集日時がサマリー最終更新より新しいページ（= 内容が変わったページ）
を対象に、各ページのプロパティ情報をClaudeに読ませてサマリーを自動生成・書き込む。

実行方法:
  python3 auto_summarize.py                   # 全DB対象
  python3 auto_summarize.py --policy-only     # 施策DBのみ（即時同期用）
  python3 auto_summarize.py --dry-run         # 書き込まずに対象ページだけ表示
  python3 auto_summarize.py --dry-run --policy-only

cron での自動実行例（毎朝8時）:
  0 8 * * * cd /path/to/Q&Ai && python3 auto_summarize.py >> logs/auto_summarize.log 2>&1
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import httpx
from dotenv import load_dotenv

# ─── .env 強制読み込み ────────────────────────────────────────
BASE_DIR          = Path(__file__).parent
LAST_SYNC_FILE    = BASE_DIR / "data" / "last_sync.txt"      # 同期完了日時の記録ファイル
SUMMARY_TS_FILE   = BASE_DIR / "data" / "summary_last_run.txt"  # マニュアル/料金表の最終サマリー生成日時
_env_file = BASE_DIR / ".env"

def _force_load_env(path: Path) -> None:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ[k.strip()] = v.strip()

_force_load_env(_env_file)

# ─── 設定 ─────────────────────────────────────────────────────
NOTION_API_KEY  = os.environ.get("NOTION_API_KEY", "")
ANTHROPIC_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
SUMMARIZE_MODEL = "claude-haiku-4-5"   # 要約はhaiku（コスト抑制）
REQUEST_INTERVAL = 0.5                  # Notion API間隔（秒）
MAX_PAGES_PER_RUN = 50                 # 1回の実行で処理する最大ページ数

DB_IDS = {
    "policy":  os.environ.get("NOTION_DB_POLICY",  ""),
    "manual":  os.environ.get("NOTION_DB_MANUAL",  ""),
    "pricing": os.environ.get("NOTION_DB_PRICING", ""),
}

NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

# ─── DB別 サマリー生成プロンプト ─────────────────────────────
PROMPTS = {
    "policy": """\
あなたはコールセンター向け社内ツールのサマリー作成者です。
以下の「施策ページのプロパティ情報」を読み、コールセンターのパートスタッフが
「この施策は今どういう状態で、どう対応すべきか」を即座に理解できるサマリーを作成してください。

【出力ルール】
- 箇条書き（・）を使い、200〜400文字以内にまとめる
- 必ず含める項目: ①状況/ステータス ②対象商品・対象者 ③施策の内容（特典・コード等） ④注意事項
- 「運用前」「運用中」「運用終了」など状況を冒頭に必ず明記する
- 推測・補足説明は不要。プロパティに記載のある情報だけを使う
- 出力はサマリー本文のみ（説明文・前置き不要）

【施策プロパティ情報】
{properties}
""",
    "manual": """\
あなたはコールセンター向け社内ツールのサマリー作成者です。
以下の「マニュアルページのプロパティ情報」を読み、コールセンターのパートスタッフが
「このマニュアルに何が書かれているか・どう対応するか」を即座に理解できるサマリーを作成してください。

【出力ルール】
- 箇条書き（・）を使い、200〜400文字以内にまとめる
- 必ず含める項目: ①ルール・手順の要点 ②例外ケース ③禁止事項（あれば）
- 推測・補足説明は不要。プロパティに記載のある情報だけを使う
- 出力はサマリー本文のみ（説明文・前置き不要）

【マニュアルプロパティ情報】
{properties}
""",
    "pricing": """\
あなたはコールセンター向け社内ツールのサマリー作成者です。
以下の「料金表ページのプロパティ情報」を読み、コールセンターのパートスタッフが
「この商品コースの価格・条件・解約ルール」を即座に理解できるサマリーを作成してください。

【出力ルール】
- 箇条書き（・）を使い、200〜400文字以内にまとめる
- 必ず含める項目: ①初回金額（税込・送料条件） ②2回目以降金額 ③配送サイクル ④解約条件・期限
- 金額は必ず「税込〇〇円（送料別）」または「税込〇〇円（送料込）」形式で記載する
- 推測・補足説明は不要。プロパティに記載のある情報だけを使う
- 出力はサマリー本文のみ（説明文・前置き不要）

【料金表プロパティ情報】
{properties}
""",
}

# ─── ユーティリティ ───────────────────────────────────────────

def get_text(prop: dict) -> str:
    """rich_text / title プロパティからテキストを取り出す"""
    ptype = prop.get("type", "")
    if ptype in ("rich_text", "title"):
        return "".join(t["plain_text"] for t in prop.get(ptype, [])).strip()
    if ptype == "formula":
        return prop.get("formula", {}).get("string", "").strip()
    return ""


def props_to_text(props: dict, db_type: str) -> str:
    """
    プロパティ辞書を「キー: 値」形式のテキストに変換する。
    サマリー生成の入力として使う。
    """
    lines = []

    # DB別に重要プロパティを優先表示
    priority = {
        "policy":  ["施策名", "状況", "対象商品", "対象者", "施策内容", "施策概要",
                    "初回コード", "初回金額", "初回サイクル", "初回配送個数",
                    "2回目以降コード", "2回目以降金額", "2回目以降サイクル", "2回目以降 配送個数",
                    "クーポンコード", "特典", "注意事項", "処理/対応", "実施期間"],
        "manual":  ["id", "カテゴリ大", "カテゴリ中", "カテゴリ小", "マニュアル"],
        "pricing": ["id", "コース名", "商品", "カテゴリ大", "カテゴリ中",
                    "初回金額", "2回目以降金額", "3回目以降金額",
                    "コース概要", "配送方法", "解約", "備考"],
    }

    ordered_keys = priority.get(db_type, [])
    shown = set()

    for key in ordered_keys:
        if key not in props:
            continue
        prop = props[key]
        ptype = prop.get("type", "")
        val = ""
        if ptype in ("rich_text", "title"):
            val = get_text(prop)
        elif ptype == "status":
            val = prop.get("status", {}).get("name", "")
        elif ptype == "select":
            val = prop.get("select", {}).get("name", "") if prop.get("select") else ""
        elif ptype == "multi_select":
            val = "、".join(o["name"] for o in prop.get("multi_select", []))
        elif ptype == "date":
            d = prop.get("date")
            val = d.get("start", "") if d else ""
        elif ptype == "number":
            n = prop.get("number")
            val = str(n) if n is not None else ""
        elif ptype == "checkbox":
            val = "はい" if prop.get("checkbox") else "いいえ"
        if val:
            lines.append(f"{key}: {val}")
            shown.add(key)

    # 残りのプロパティも追加（メタ系は除外）
    skip_types = {"created_by", "people", "files", "last_edited_time",
                  "created_time", "rollup", "relation", "formula"}
    skip_keys  = {"サマリー", "サマリー最終更新", "サマリー更新対象", "サマリー更新中",
                  "更新‼️", "バナー/同梱物", "作成者", "作成者 ", "社外共有", "社内共有"}
    for key, prop in props.items():
        if key in shown or key in skip_keys:
            continue
        if prop.get("type") in skip_types:
            continue
        ptype = prop.get("type", "")
        val = ""
        if ptype in ("rich_text", "title"):
            val = get_text(prop)
        elif ptype == "select":
            val = prop.get("select", {}).get("name", "") if prop.get("select") else ""
        elif ptype == "multi_select":
            val = "、".join(o["name"] for o in prop.get("multi_select", []))
        elif ptype == "status":
            val = prop.get("status", {}).get("name", "")
        elif ptype == "number":
            n = prop.get("number")
            val = str(n) if n is not None else ""
        elif ptype == "checkbox":
            val = "はい" if prop.get("checkbox") else ""
        if val:
            lines.append(f"{key}: {val}")

    return "\n".join(lines) if lines else "（プロパティ情報なし）"


def get_title(props: dict) -> str:
    """ページタイトルを返す"""
    for key in ["施策名", "名前", "Name", "タイトル", "Title", "id"]:
        if key in props:
            t = get_text(props[key])
            if t:
                return t
    for val in props.values():
        if val.get("type") == "title":
            t = get_text(val)
            if t:
                return t
    return "（タイトル不明）"


def parse_iso(s: str) -> datetime:
    """ISO8601文字列をUTC aware datetimeに変換"""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


# ─── Notion API ──────────────────────────────────────────────

def fetch_all_pages(db_id: str) -> list[dict]:
    """DBの全ページをページネーションで取得"""
    pages = []
    cursor = None
    while True:
        body = {"page_size": 100}
        if cursor:
            body["start_cursor"] = cursor
        resp = httpx.post(
            f"https://api.notion.com/v1/databases/{db_id}/query",
            headers=NOTION_HEADERS,
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        pages.extend(data.get("results", []))
        if not data.get("has_more"):
            break
        cursor = data["next_cursor"]
        time.sleep(REQUEST_INTERVAL)
    return pages


def write_summary_to_notion(page_id: str, summary: str, dry_run: bool = False) -> bool:
    """NotionページのサマリープロパティとサマリーDB更新日時を書き込む"""
    if dry_run:
        return True
    today = datetime.now().strftime("%Y-%m-%d")
    payload = {
        "properties": {
            "サマリー": {
                "rich_text": [{"text": {"content": summary[:2000]}}]
            },
        }
    }
    # 施策DBのみ「サマリー最終更新」日付プロパティを更新
    # （他DBにはこのプロパティが存在しないため条件付き）
    try:
        resp = httpx.patch(
            f"https://api.notion.com/v1/pages/{page_id}",
            headers=NOTION_HEADERS,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"    ⚠ Notion書き込みエラー: {e}")
        return False


def update_summary_date(page_id: str, dry_run: bool = False) -> None:
    """施策DBの「サマリー最終更新」日付プロパティを今日の日付で更新"""
    if dry_run:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        resp = httpx.patch(
            f"https://api.notion.com/v1/pages/{page_id}",
            headers=NOTION_HEADERS,
            json={"properties": {"サマリー最終更新": {"date": {"start": today}}}},
            timeout=30,
        )
        resp.raise_for_status()
    except Exception:
        pass   # 日付更新の失敗はサイレントスキップ


def write_last_sync(policy_only: bool = False, dry_run: bool = False) -> None:
    """同期完了日時を last_sync.txt に書き込む"""
    if dry_run:
        return
    LAST_SYNC_FILE.parent.mkdir(exist_ok=True)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scope   = "施策のみ" if policy_only else "全DB"
    with open(LAST_SYNC_FILE, "w", encoding="utf-8") as f:
        f.write(f"{now_str}\t{scope}\n")
    print(f"  最終同期日時を記録: {LAST_SYNC_FILE}")


def read_summary_last_run():
    """マニュアル/料金表の前回サマリー生成日時を返す。なければ None。"""
    if not SUMMARY_TS_FILE.exists():
        return None
    try:
        s = SUMMARY_TS_FILE.read_text(encoding="utf-8").strip()
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def write_summary_last_run(dry_run: bool = False) -> None:
    """マニュアル/料金表のサマリー生成完了日時を UTC ISO 形式で保存する。"""
    if dry_run:
        return
    SUMMARY_TS_FILE.parent.mkdir(exist_ok=True)
    now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    SUMMARY_TS_FILE.write_text(now_utc, encoding="utf-8")
    print(f"  マニュアル/料金表 前回実行日時を記録: {now_utc}")


# ─── サマリー生成 ─────────────────────────────────────────────

def generate_summary(props_text: str, db_type: str, client: anthropic.Anthropic) -> str:
    """Claudeを使ってサマリーを生成する"""
    prompt = PROMPTS[db_type].format(properties=props_text)
    resp = client.messages.create(
        model=SUMMARIZE_MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


# ─── 対象判定 ─────────────────────────────────────────────────

def needs_update(page: dict, db_type: str) -> tuple[bool, str]:
    """
    サマリーを生成・更新すべきかを判定する。
    戻り値: (対象かどうか, 理由)
    """
    props = page.get("properties", {})

    # ① サマリーが空欄
    current_summary = get_text(props.get("サマリー", {}))
    if not current_summary:
        return True, "サマリー空欄"

    # ② 施策・マニュアル・料金表: ページの最終編集日時 > 前回サマリー生成日時 なら更新対象
    if db_type in ("policy", "manual", "pricing"):
        # 「最終更新日時」プロパティ（last_edited_time 型）を優先、なければページ本体の last_edited_time
        last_edited_prop = props.get("最終更新日時", {})
        last_edited_str  = (last_edited_prop.get("last_edited_time", "")
                            or page.get("last_edited_time", ""))
        if last_edited_str:
            last_edited    = parse_iso(last_edited_str)
            prev_run       = read_summary_last_run()
            if prev_run is None:
                # 初回実行（前回記録なし）→ 全件対象
                return True, "初回実行（前回記録なし）"
            if last_edited > prev_run:
                return True, f"ページ更新({last_edited_str[:16]}) > 前回サマリー生成({prev_run.isoformat()[:16]})"

    return False, ""


# ─── メイン処理 ──────────────────────────────────────────────

def process_db(db_type: str, db_id: str, client: anthropic.Anthropic,
               dry_run: bool, stats: dict) -> None:
    label = {"policy": "施策", "manual": "マニュアル", "pricing": "料金表"}[db_type]
    print(f"\n{'='*50}")
    print(f"  {label} DB を処理中...")
    print(f"{'='*50}")

    if not db_id:
        print(f"  [SKIP] DB IDが未設定")
        return

    pages = fetch_all_pages(db_id)
    print(f"  取得: {len(pages)} 件")

    targets = []
    for page in pages:
        ok, reason = needs_update(page, db_type)
        if ok:
            targets.append((page, reason))

    print(f"  更新対象: {len(targets)} 件")
    if not targets:
        print("  ✅ 全ページのサマリーは最新です")
        return

    processed = 0
    for page, reason in targets:
        if processed >= MAX_PAGES_PER_RUN:
            print(f"\n  ⚠ 上限({MAX_PAGES_PER_RUN}件)に達したため残りは次回実行時に処理されます")
            break

        props = page.get("properties", {})
        title = get_title(props)
        page_id = page["id"]

        print(f"\n  [{processed+1}/{len(targets)}] {title}")
        print(f"    理由: {reason}")

        props_text = props_to_text(props, db_type)

        try:
            summary = generate_summary(props_text, db_type, client)
            print(f"    生成: {summary[:80].replace(chr(10),' ')}…")

            if write_summary_to_notion(page_id, summary, dry_run):
                if db_type == "policy" and not dry_run:
                    update_summary_date(page_id)
                status = "✅ 書き込み完了" if not dry_run else "🔍 [dry-run] 書き込みスキップ"
                print(f"    {status}")
                stats["success"] += 1
            else:
                stats["error"] += 1

        except Exception as e:
            print(f"    ❌ 生成エラー: {e}")
            stats["error"] += 1

        processed += 1
        stats["total"] += 1
        time.sleep(REQUEST_INTERVAL)

    # 処理完了後に「前回実行日時」を記録する（次回起動時の比較基準になる）
    if not dry_run and stats["success"] > 0:
        write_summary_last_run(dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(description="Notionサマリー自動生成ツール")
    parser.add_argument("--dry-run", action="store_true",
                        help="書き込みを行わず、対象ページと生成サマリーの確認だけ行う")
    parser.add_argument("--db", choices=["policy", "manual", "pricing"],
                        help="特定のDBだけ処理する（--policy-only の代替）")
    parser.add_argument("--policy-only", action="store_true",
                        help="施策DBのみを対象に処理する（即時同期用）")
    args = parser.parse_args()

    # --policy-only は --db policy と同等
    if args.policy_only:
        args.db = "policy"

    if not NOTION_API_KEY:
        print("[ERROR] NOTION_API_KEY が未設定です")
        sys.exit(1)
    if not ANTHROPIC_KEY:
        print("[ERROR] ANTHROPIC_API_KEY が未設定です")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    print("=" * 50)
    print("  Notionサマリー自動生成ツール")
    if args.policy_only or args.db == "policy":
        print("  ⚡ 施策DBのみモード（即時同期）")
    if args.dry_run:
        print("  ⚠ DRY-RUN モード（Notionへの書き込みは行いません）")
    print(f"  実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    stats = {"total": 0, "success": 0, "error": 0}

    db_targets = (
        [(args.db, DB_IDS[args.db])]
        if args.db
        else [("policy", DB_IDS["policy"]),
              ("manual", DB_IDS["manual"]),
              ("pricing", DB_IDS["pricing"])]
    )

    for db_type, db_id in db_targets:
        process_db(db_type, db_id, client, args.dry_run, stats)

    # 同期完了日時を記録
    policy_only_flag = bool(args.db == "policy" or args.policy_only)
    write_last_sync(policy_only=policy_only_flag, dry_run=args.dry_run)

    print(f"\n{'='*50}")
    print(f"  完了サマリー")
    print(f"  処理件数: {stats['total']} 件")
    print(f"  ✅ 成功: {stats['success']} 件")
    print(f"  ❌ エラー: {stats['error']} 件")
    print(f"{'='*50}")

    if not args.dry_run and stats["success"] > 0:
        print("\n次のステップ: python3 fetch_knowledge.py --policy-only を実行して知識ベースを更新してください。")


if __name__ == "__main__":
    main()
