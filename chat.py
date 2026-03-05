"""
chat.py
ターミナルで動作する最小構成のチャットプロトタイプ。
data/knowledge.json を読み込み、Claude API で回答する。

実行方法:
  python3 chat.py
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DATA_DIR     = Path(__file__).parent / "data"
LOGS_DIR     = Path(__file__).parent / "logs"
KNOWLEDGE_JSON = DATA_DIR / "knowledge.json"
LOG_FILE       = LOGS_DIR / "chat_log.json"

MAX_KNOWLEDGE_CHARS = 80_000   # システムプロンプトに含める知識の上限文字数


# ---------- 知識ベース読み込み ----------

def load_knowledge() -> str:
    """knowledge.json からテキスト形式の知識を構築する"""
    if not KNOWLEDGE_JSON.exists():
        return "（知識ファイルがありません。先に fetch_knowledge.py を実行してください）"

    with open(KNOWLEDGE_JSON, encoding="utf-8") as f:
        data = json.load(f)

    fetched_at = data.get("fetched_at", "不明")
    sections = []

    section_meta = [
        ("policy",  "【施策】（最優先：特例・キャンペーン情報）"),
        ("manual",  "【マニュアル】（ルール・手順）"),
        ("pricing", "【料金表】（価格・プラン情報）"),
    ]

    for key, header in section_meta:
        records = data.get(key, [])
        if not records:
            continue
        block = [f"## {header}"]
        for r in records:
            block.append(f"### {r['title']}")
            block.append(r["summary"])
        sections.append("\n".join(block))

    body = "\n\n".join(sections)
    # 文字数制限
    if len(body) > MAX_KNOWLEDGE_CHARS:
        body = body[:MAX_KNOWLEDGE_CHARS] + "\n\n（文字数制限により以降省略）"

    return f"知識ベース更新日時: {fetched_at}\n優先順位: 施策 > マニュアル > 料金表\n\n{body}"


SYSTEM_PROMPT_TEMPLATE = """\
あなたは通販コールセンターのAIアシスタントです。
パートスタッフの質問に、正確・簡潔・丁寧に回答してください。

## 回答ルール
1. 以下の知識ベースを必ず参照し、「施策 → マニュアル → 料金表」の優先順位で回答する。
2. マニュアルと施策で情報が矛盾する場合は、まずマニュアルの標準ルールを回答し、その後に「ただし、現在〇〇の施策が適用されている場合は例外となります」と添える。
3. 知識ベースに情報がない場合は「確認が必要です」と伝え、推測で答えない。
4. 回答は箇条書きを活用し、要点を先に述べる。

## 知識ベース
{knowledge}
"""


# ---------- ログ ----------

def load_log() -> list:
    LOGS_DIR.mkdir(exist_ok=True)
    if LOG_FILE.exists():
        with open(LOG_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_log(log: list):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def append_log(log: list, entry: dict):
    log.append(entry)
    save_log(log)


# ---------- チャット ----------

def chat(question: str, knowledge: str, client: anthropic.Anthropic) -> str:
    system = SYSTEM_PROMPT_TEMPLATE.format(knowledge=knowledge)
    resp = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    return resp.content[0].text


def ask_feedback():
    """解決した/しない を入力させる"""
    while True:
        ans = input("  解決しましたか？ [y/n/skip] > ").strip().lower()
        if ans in ("y", "yes"):
            return "resolved"
        if ans in ("n", "no"):
            return "unresolved"
        if ans in ("s", "skip", ""):
            return None
        print("  y / n / skip で入力してください")


def main():
    if not ANTHROPIC_API_KEY:
        print("[ERROR] .env に ANTHROPIC_API_KEY が設定されていません")
        return

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    knowledge = load_knowledge()
    log = load_log()

    print("=" * 50)
    print("  コールセンター AIアシスタント（Ctrl+C で終了）")
    print("=" * 50)

    while True:
        try:
            question = input("\n質問> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n終了します")
            break

        if not question:
            continue

        print("\n回答を生成中...\n")
        answer = chat(question, knowledge, client)
        print(f"回答:\n{answer}\n")

        feedback = ask_feedback()

        entry = {
            "id":         str(uuid.uuid4()),
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            "question":   question,
            "answer":     answer,
            "feedback":   feedback,
        }
        append_log(log, entry)


if __name__ == "__main__":
    main()
