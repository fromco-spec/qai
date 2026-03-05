"""
update_notion_manual.py

updated_manual.csv を読み込み、Notion マニュアルDBの「サマリー」プロパティを
一括上書きするスクリプト。

使い方:
  # ドライラン（Notion API 呼び出しなし・CSV内容の確認のみ）
  python3 update_notion_manual.py --dry-run

  # 実際に更新
  python3 update_notion_manual.py

前提:
  - .env に NOTION_API_KEY / NOTION_DB_MANUAL が設定済み
  - updated_manual.csv が /tmp/cs_manual2/ に存在する
  - pip install notion-client でインストール済み
"""

import argparse
import csv
import sys
import time
from pathlib import Path

# ---------- 設定 ----------

BASE_DIR   = Path(__file__).parent
CSV_PATH   = Path("/tmp/cs_manual2/updated_manual.csv")
ENV_FILE   = BASE_DIR / ".env"

# Notion API レート制限対策: リクエスト間待機秒数（上限 ~3 req/sec）
REQUEST_INTERVAL = 0.35


# ---------- .env 読み込み ----------

def load_env(path: Path) -> dict:
    env = {}
    if not path.exists():
        return env
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                env[key.strip()] = val.strip()
    return env


# ---------- Notion DBクエリ: id文字列 → page_id マッピング ----------

def build_id_to_pageid(client, db_id: str) -> dict:
    """
    search API でマニュアルDB配下のページを全件取得し、
    タイトル(manual-XXX) → Notion page_id の辞書を返す。

    notion-client v2.7.0 では databases/query エンドポイントが廃止されたため、
    search API を使って親DBでフィルタリングする方式に変更。
    """
    # ハイフンなしIDに正規化（比較用）
    db_id_plain = db_id.replace("-", "")

    mapping = {}
    cursor = None

    print("Notion DBからページ一覧を取得中（search API使用）...", flush=True)
    while True:
        kwargs = {
            "filter": {"value": "page", "property": "object"},
            "page_size": 100,
        }
        if cursor:
            kwargs["start_cursor"] = cursor

        resp = client.search(**kwargs)

        for page in resp.get("results", []):
            # 親がマニュアルDBのページだけを対象にする
            parent = page.get("parent", {})
            parent_db_id = parent.get("database_id", "").replace("-", "")
            if parent_db_id != db_id_plain:
                continue

            page_id = page["id"]
            # title 型のプロパティを動的に探す
            for _prop_name, prop_val in page.get("properties", {}).items():
                if prop_val.get("type") == "title":
                    texts = prop_val["title"]
                    if texts:
                        title_str = "".join(t["plain_text"] for t in texts)
                        mapping[title_str] = page_id
                    break

        if resp.get("has_more"):
            cursor = resp.get("next_cursor")
        else:
            break

        time.sleep(REQUEST_INTERVAL)

    print(f"  → {len(mapping)} 件取得", flush=True)
    return mapping


# ---------- サマリー更新 ----------

def update_summary(client, page_id: str, new_summary: str):
    """
    指定ページの「サマリー」rich_text プロパティを上書き。
    Notion rich_text は1要素あたり2000文字上限のため、必要に応じて分割。
    """
    CHUNK_SIZE = 2000
    chunks = [new_summary[i:i + CHUNK_SIZE]
              for i in range(0, len(new_summary), CHUNK_SIZE)]
    rich_text = [{"type": "text", "text": {"content": c}} for c in chunks]

    client.pages.update(
        page_id=page_id,
        properties={
            "サマリー": {
                "rich_text": rich_text
            }
        }
    )


# ---------- メイン ----------

def main():
    parser = argparse.ArgumentParser(description="Notion マニュアルDB サマリー一括更新")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Notion API を呼ばずに CSV の内容だけ確認する"
    )
    args = parser.parse_args()

    # .env 読み込み
    env = load_env(ENV_FILE)
    notion_api_key = env.get("NOTION_API_KEY", "")
    db_manual_id   = env.get("NOTION_DB_MANUAL", "")

    # CSV 読み込み
    if not CSV_PATH.exists():
        print(f"[ERROR] CSVが見つかりません: {CSV_PATH}")
        sys.exit(1)

    with open(CSV_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"CSV読み込み: {len(rows)} 行")

    # ========== ドライラン ==========
    if args.dry_run:
        print("\n[DRY-RUN] Notion API は呼び出しません。CSV 内容のプレビューを表示します。\n")
        for i, row in enumerate(rows, 1):
            record_id   = row.get("id", "").strip()
            new_summary = row.get("サマリー", "").strip()
            tag_start   = new_summary.rfind("【検索タグ：")
            tag_part    = new_summary[tag_start:] if tag_start >= 0 else "（タグなし）"
            preview     = new_summary[:60].replace("\n", "\\n")
            print(f"  [{i:3d}] {record_id:12s}  先頭: {preview}...")
            print(f"         {tag_part}")
        print(f"\n[DRY-RUN完了] {len(rows)} 行を確認しました。")
        print("実際に更新するには --dry-run を外して再実行してください:")
        print("  python3 update_notion_manual.py")
        return

    # ========== 実行 ==========
    if not notion_api_key:
        print("[ERROR] .env に NOTION_API_KEY が設定されていません")
        sys.exit(1)
    if not db_manual_id:
        print("[ERROR] .env に NOTION_DB_MANUAL が設定されていません")
        sys.exit(1)

    from notion_client import Client
    client = Client(auth=notion_api_key)

    # id → page_id マッピング取得
    id_to_pageid = build_id_to_pageid(client, db_manual_id)

    success = 0
    skipped = 0
    errors  = 0

    print("\n[更新] サマリー更新開始...\n")

    for i, row in enumerate(rows, 1):
        record_id   = row.get("id", "").strip()
        new_summary = row.get("サマリー", "").strip()

        if not record_id:
            print(f"  [{i:3d}] IDなし → スキップ")
            skipped += 1
            continue

        page_id = id_to_pageid.get(record_id)
        if not page_id:
            print(f"  [{i:3d}] {record_id} → Notionに該当ページなし → スキップ")
            skipped += 1
            continue

        try:
            update_summary(client, page_id, new_summary)
            print(f"  [{i:3d}] {record_id} → OK")
            success += 1
        except Exception as e:
            print(f"  [{i:3d}] {record_id} → ERROR: {e}")
            errors += 1

        time.sleep(REQUEST_INTERVAL)

    print(f"\n{'='*40}")
    print("更新完了")
    print(f"  成功:     {success} 件")
    print(f"  スキップ:   {skipped} 件")
    print(f"  エラー:   {errors} 件")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
