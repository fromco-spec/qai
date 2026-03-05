"""
run_test.py
evaluation_problem_set.csv の全質問を RAG+Claude パイプラインに流し、
自動採点して test_results.csv に書き出す。
※ Streamlit に依存せず単独で動作する完全独立版。

実行方法:
  cd /path/to/Q&Ai
  python3 run_test.py
"""

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

# ─── .env 読み込み（override=True で確実上書き） ──────────────
BASE_DIR = Path(__file__).parent
_env_file = BASE_DIR / ".env"

def _load_env(path: Path):
    """最小限の .env パーサー"""
    env = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
    return env

_env = _load_env(_env_file)
for k, v in _env.items():
    os.environ.setdefault(k, v)
# override で上書き
for k, v in _env.items():
    os.environ[k] = v

import anthropic

# ─── パス設定 ────────────────────────────────────────────────
EVAL_CSV       = BASE_DIR / "evaluation_problem_set.csv"
RESULT_CSV     = BASE_DIR / "test_results.csv"
KNOWLEDGE_JSON = BASE_DIR / "data" / "knowledge.json"

# ─── モデル設定 ─────────────────────────────────────────────
ANSWER_MODEL     = "claude-opus-4-5-20251101"
JUDGE_MODEL      = "claude-haiku-4-5"
REQUEST_INTERVAL = 1.0

# ─── RAG設定（app.py と同じ値） ───────────────────────────────
RAG_TOP_POLICY  = 20
RAG_TOP_MANUAL  = 30
RAG_TOP_PRICING = 10

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

# ─── 知識ベース読み込み ──────────────────────────────────────
def load_knowledge() -> dict:
    if not KNOWLEDGE_JSON.exists():
        print(f"[ERROR] {KNOWLEDGE_JSON} が見つかりません。fetch_knowledge.py を先に実行してください。")
        sys.exit(1)
    with open(KNOWLEDGE_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "policy":  data.get("policy",  []),
        "manual":  data.get("manual",  []),
        "pricing": data.get("pricing", []),
    }


# ─── RAGコア（app.pyと同ロジック・Streamlit非依存） ────────────

def _tokenize(text: str) -> set:
    tokens = re.findall(r'[A-Za-z0-9０-９Ａ-Ｚａ-ｚ\u3040-\u9fff\u30a0-\u30ff]+', text)
    result = set()
    for t in tokens:
        if len(t) >= 2:
            result.add(t.lower())
            for i in range(len(t) - 1):
                result.add(t[i:i+2].lower())
    return result


def _clean_summary(text: str) -> str:
    text = re.sub(r'^:::[ \t]+callout[^\n]*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'^:::[ \t]*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _score(record: dict, query_tokens: set) -> int:
    target = (record.get("title", "") + " " + record.get("summary", "")).lower()
    return len(query_tokens & _tokenize(target))


def _expand_query(question: str, client: anthropic.Anthropic) -> str:
    try:
        resp = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=200,
            system=(
                "あなたはコールセンター向け社内マニュアル検索の専門家です。\n"
                "ユーザーの質問文を受け取り、検索ヒット率を上げるための"
                "同義語・関連語・上位概念語を追加した拡張クエリを返してください。\n"
                "出力形式：元の質問に続けて関連語をスペース区切りで追記した1行のテキストのみ。\n"
                "余計な説明・記号・改行は不要。"
            ),
            messages=[{"role": "user", "content": question}],
        )
        expanded = resp.content[0].text.strip()
        return expanded if expanded else question
    except Exception:
        return question


def retrieve_knowledge_standalone(question: str, records: dict, client: anthropic.Anthropic) -> str:
    expanded_query = _expand_query(question, client)
    query_tokens   = _tokenize(expanded_query)

    section_meta = [
        ("policy",  "【施策】（最優先：特例・キャンペーン情報）",  RAG_TOP_POLICY),
        ("manual",  "【マニュアル】（ルール・手順）",             RAG_TOP_MANUAL),
        ("pricing", "【料金表】（価格・プラン情報）",             RAG_TOP_PRICING),
    ]
    sections = []
    retrieved_counts = {}
    for key, header, top_n in section_meta:
        all_recs = records.get(key, [])
        if not all_recs:
            continue
        scored   = sorted(all_recs, key=lambda r: _score(r, query_tokens), reverse=True)
        selected = scored[:top_n]
        block = [f"## {header}"]
        for r in selected:
            block.append(f"### {r['title']}\n{_clean_summary(r['summary'])}")
        sections.append("\n".join(block))
        retrieved_counts[key] = len(selected)

    body = "\n\n".join(sections)
    summary_line = (
        f"[RAG: 施策{retrieved_counts.get('policy',0)}件 / "
        f"マニュアル{retrieved_counts.get('manual',0)}件 / "
        f"料金表{retrieved_counts.get('pricing',0)}件 / "
        f"拡張: {expanded_query[:60]}{'…' if len(expanded_query)>60 else ''}]"
    )
    return f"{summary_line}\n優先順位: 施策 > マニュアル > 料金表\n\n{body}"


def get_answer_standalone(question: str, knowledge: str, client: anthropic.Anthropic) -> str:
    system = SYSTEM_PROMPT_TEMPLATE.format(knowledge=knowledge)
    resp = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    return resp.content[0].text


# ─── 自動採点 ────────────────────────────────────────────────

def judge_answer(question: str, expected: str, actual: str, client: anthropic.Anthropic) -> tuple[str, str]:
    system = (
        "あなたはコールセンター向けAIの評価者です。\n"
        "以下の「質問」「期待される回答のポイント」「実際の回答」を比較し、"
        "実際の回答が期待ポイントをどの程度カバーしているか評価してください。\n\n"
        "採点基準:\n"
        "- PASS   : 期待ポイントの主要事項を正確に含んでいる\n"
        "- PARTIAL: 期待ポイントを部分的にしか含んでいない、または曖昧\n"
        "- FAIL   : 期待ポイントを含まない、または誤った情報を回答している\n\n"
        "出力形式（JSONのみ）:\n"
        '{"verdict": "PASS"|"PARTIAL"|"FAIL", "reason": "50字以内の理由"}'
    )
    user_content = (
        f"【質問】\n{question}\n\n"
        f"【期待される回答のポイント】\n{expected}\n\n"
        f"【実際の回答】\n{actual}"
    )
    try:
        resp = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=200,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = resp.content[0].text.strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            return obj.get("verdict", "FAIL"), obj.get("reason", "")
        return "FAIL", f"採点パース失敗: {raw[:80]}"
    except Exception as e:
        return "FAIL", f"採点エラー: {e}"


# ─── メイン ─────────────────────────────────────────────────

def load_eval_csv() -> list[dict]:
    if not EVAL_CSV.exists():
        print(f"[ERROR] {EVAL_CSV} が見つかりません。")
        sys.exit(1)
    rows = []
    with open(EVAL_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def run():
    print("=" * 62)
    print("  コールセンターAI 一括評価テスト（Streamlit独立版）")
    print("=" * 62)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[ERROR] ANTHROPIC_API_KEY が取得できません。.env を確認してください。")
        sys.exit(1)
    print(f"APIキー確認: {api_key[:20]}…\n")

    client   = anthropic.Anthropic(api_key=api_key)
    records  = load_knowledge()
    problems = load_eval_csv()
    print(f"問題数: {len(problems)} 件\n")

    results = []
    pass_count = partial_count = fail_count = 0

    for idx, prob in enumerate(problems, 1):
        q_id     = prob.get("id", f"Q{idx}")
        category = prob.get("category", "")
        question = prob.get("question", "")
        expected = prob.get("expected_points", "")

        print(f"[{idx:02d}/{len(problems)}] {q_id} ({category})")
        print(f"  質問: {question}")

        # Step1: RAG
        try:
            knowledge = retrieve_knowledge_standalone(question, records, client)
        except Exception as e:
            knowledge = ""
            print(f"  ⚠ RAGエラー: {e}")
        time.sleep(REQUEST_INTERVAL)

        # Step2: 回答生成
        try:
            actual = get_answer_standalone(question, knowledge, client)
        except Exception as e:
            actual = f"[ERROR] {e}"
            print(f"  ⚠ 回答生成エラー: {e}")
        time.sleep(REQUEST_INTERVAL)

        # Step3: 採点
        verdict, reason = judge_answer(question, expected, actual, client)
        time.sleep(REQUEST_INTERVAL)

        if verdict == "PASS":
            pass_count += 1; mark = "✅"
        elif verdict == "PARTIAL":
            partial_count += 1; mark = "⚠️"
        else:
            fail_count += 1; mark = "❌"

        # 回答の先頭100文字だけ表示
        answer_preview = actual.replace("\n", " ")[:120]
        print(f"  回答: {answer_preview}…")
        print(f"  {mark} {verdict}  ← {reason}\n")

        results.append({
            "id":              q_id,
            "category":        category,
            "question":        question,
            "expected_points": expected,
            "actual_answer":   actual,
            "verdict":         verdict,
            "reason":          reason,
        })

    # ── 集計 ──
    total        = len(results)
    pass_rate    = pass_count    / total * 100
    partial_rate = partial_count / total * 100
    fail_rate    = fail_count    / total * 100
    score        = (pass_count + partial_count * 0.5) / total * 100

    print("=" * 62)
    print(f"  テスト完了: {total} 問")
    print(f"  ✅ PASS   : {pass_count:3d} 件  ({pass_rate:.1f}%)")
    print(f"  ⚠️  PARTIAL: {partial_count:3d} 件  ({partial_rate:.1f}%)")
    print(f"  ❌ FAIL   : {fail_count:3d} 件  ({fail_rate:.1f}%)")
    print(f"  ────────────────────────────────")
    print(f"  📊 総合スコア (PASS=1点 / PARTIAL=0.5点): {score:.1f}%")
    print("=" * 62)

    # ── CSV出力 ──
    fieldnames = ["id","category","question","expected_points",
                  "actual_answer","verdict","reason"]
    with open(RESULT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    with open(RESULT_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({
            "id": "SUMMARY",
            "category": f"PASS={pass_count} PARTIAL={partial_count} FAIL={fail_count}",
            "question": f"総合スコア: {score:.1f}%",
            "expected_points": "",
            "actual_answer": "",
            "verdict": "",
            "reason": f"PASS率={pass_rate:.1f}% PARTIAL率={partial_rate:.1f}% FAIL率={fail_rate:.1f}%",
        })

    print(f"\n結果保存: {RESULT_CSV}")
    return score, pass_count, partial_count, fail_count


if __name__ == "__main__":
    run()
