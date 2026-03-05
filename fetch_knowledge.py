"""
fetch_knowledge.py
Notionの3つのDB（施策・マニュアル・料金表）からサマリーを取得し、
data/knowledge.json と data/knowledge.txt にキャッシュする。

実行方法:
  python3 fetch_knowledge.py                  # 全DB取得
  python3 fetch_knowledge.py --policy-only    # 施策DBのみ更新（即時同期用）

cron での自動実行例（1時間ごと）:
  0 * * * * cd /path/to/Q&Ai && python3 fetch_knowledge.py
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

# .env 強制読み込み
def _force_load_env(path: Path) -> None:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ[k.strip()] = v.strip()

BASE_DIR = Path(__file__).parent
_force_load_env(BASE_DIR / ".env")

NOTION_API_KEY = os.environ.get("NOTION_API_KEY", "")
DB_IDS = {
    "policy":  os.environ.get("NOTION_DB_POLICY",  ""),  # 施策（最優先）
    "manual":  os.environ.get("NOTION_DB_MANUAL",  ""),  # マニュアル
    "pricing": os.environ.get("NOTION_DB_PRICING", ""),  # 料金表
}

SUMMARY_PROP = "サマリー"
TITLE_PROPS  = ["名前", "Name", "タイトル", "Title"]

DATA_DIR       = BASE_DIR / "data"
KNOWLEDGE_JSON = DATA_DIR / "knowledge.json"
KNOWLEDGE_TXT  = DATA_DIR / "knowledge.txt"
LAST_SYNC_FILE = DATA_DIR / "last_sync.txt"


def get_title(props: dict) -> str:
    for key in TITLE_PROPS:
        if key in props:
            titles = props[key].get("title", [])
            return "".join(t["plain_text"] for t in titles).strip()
    for val in props.values():
        if val.get("type") == "title":
            return "".join(t["plain_text"] for t in val.get("title", [])).strip()
    return "（タイトル不明）"


def get_summary(props: dict) -> str:
    if SUMMARY_PROP not in props:
        return ""
    prop = props[SUMMARY_PROP]
    ptype = prop.get("type", "")
    if ptype == "rich_text":
        return "".join(t["plain_text"] for t in prop.get("rich_text", [])).strip()
    if ptype == "formula":
        return prop.get("formula", {}).get("string", "").strip()
    return ""


def fetch_db(headers: dict, db_id: str, label: str) -> list[dict]:
    if not db_id:
        print(f"  [SKIP] {label}: DB IDが未設定")
        return []

    records = []
    cursor = None
    while True:
        body = {"page_size": 100}
        if cursor:
            body["start_cursor"] = cursor
        resp = httpx.post(
            f"https://api.notion.com/v1/databases/{db_id}/query",
            headers=headers,
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        for page in data.get("results", []):
            props = page.get("properties", {})
            title   = get_title(props)
            summary = get_summary(props)
            if summary:
                records.append({
                    "label":   label,
                    "id":      page["id"],
                    "title":   title,
                    "summary": summary,
                    "url":     page.get("url", ""),
                })
        if not data.get("has_more"):
            break
        cursor = data["next_cursor"]

    print(f"  [{label}] {len(records)} 件取得")
    return records


def build_txt(knowledge: dict) -> str:
    lines = [
        "# コールセンター向け知識ベース",
        f"# 生成日時: {knowledge['fetched_at']}",
        "# 優先順位: 施策 > マニュアル > 料金表",
        "",
    ]
    for key, header in [
        ("policy",  "【施策】（最優先：特例・キャンペーン情報）"),
        ("manual",  "【マニュアル】（ルール・手順）"),
        ("pricing", "【料金表】（価格・プラン情報）"),
    ]:
        records = knowledge.get(key, [])
        lines.append(f"## {header}")
        if not records:
            lines.append("（データなし）")
        else:
            for r in records:
                lines.append(f"### {r['title']}")
                lines.append(r["summary"])
        lines.append("")
    return "\n".join(lines)


def write_last_sync(policy_only: bool = False) -> None:
    """同期完了日時を last_sync.txt に書き込む（app.py のUIが参照する）"""
    DATA_DIR.mkdir(exist_ok=True)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scope   = "施策のみ" if policy_only else "全DB"
    with open(LAST_SYNC_FILE, "w", encoding="utf-8") as f:
        f.write(f"{now_str}\t{scope}\n")
    print(f"  最終同期日時を記録: {now_str}（{scope}）")


def main():
    parser = argparse.ArgumentParser(description="Notion知識ベース取得ツール")
    parser.add_argument("--policy-only", action="store_true",
                        help="施策DBのみ更新し、マニュアル・料金表は既存データを保持する")
    args = parser.parse_args()

    if not NOTION_API_KEY:
        raise ValueError(".env に NOTION_API_KEY が設定されていません")

    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }
    DATA_DIR.mkdir(exist_ok=True)

    if args.policy_only:
        # ── 施策のみ更新モード ──────────────────────────────
        print("⚡ 施策DBのみ更新モード（即時同期）")

        # 既存の knowledge.json を読み込んで施策だけ差し替える
        existing = {}
        if KNOWLEDGE_JSON.exists():
            with open(KNOWLEDGE_JSON, encoding="utf-8") as f:
                existing = json.load(f)

        print("Notionから施策データを取得中...")
        new_policy = fetch_db(headers, DB_IDS["policy"], "施策")

        knowledge = {
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "policy":  new_policy,
            "manual":  existing.get("manual",  []),   # 既存データを保持
            "pricing": existing.get("pricing", []),   # 既存データを保持
        }
        print(f"  施策: {len(new_policy)} 件 / マニュアル・料金表: 既存データを継続使用")

    else:
        # ── 全DB更新モード ──────────────────────────────────
        print("Notionからデータを取得中...")
        knowledge = {
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "policy":  fetch_db(headers, DB_IDS["policy"],  "施策"),
            "manual":  fetch_db(headers, DB_IDS["manual"],  "マニュアル"),
            "pricing": fetch_db(headers, DB_IDS["pricing"], "料金表"),
        }

    total = sum(len(knowledge[k]) for k in ("policy", "manual", "pricing"))
    print(f"合計 {total} 件")

    # JSON保存
    with open(KNOWLEDGE_JSON, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=2)
    print(f"保存: {KNOWLEDGE_JSON}")

    # テキスト保存（デバッグ・確認用）
    with open(KNOWLEDGE_TXT, "w", encoding="utf-8") as f:
        f.write(build_txt(knowledge))
    print(f"保存: {KNOWLEDGE_TXT}")

    # 同期完了日時を記録（UIボタンが参照する）
    write_last_sync(policy_only=args.policy_only)


if __name__ == "__main__":
    main()
