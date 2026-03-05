"""
app.py
Streamlit製チャット＆管理画面。

実行方法:
  streamlit run app.py
"""

import csv
import json
import os
import re
import subprocess
import sys
import uuid
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import anthropic
import streamlit as st
from dotenv import load_dotenv
from notion_client import Client as NotionClient

# load_dotenv() だけでは既存環境変数を上書きしないケースがあるため、
# 最小限の自前パーサーで .env を強制的に os.environ へ反映する
def _force_load_env(env_path: Path) -> None:
    """override=True 相当の .env 読み込み（dotenv に依存しない）"""
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ[_k.strip()] = _v.strip()

_force_load_env(Path(__file__).parent / ".env")

# ---------- パス・定数 ----------
BASE_DIR          = Path(__file__).parent
DATA_DIR          = BASE_DIR / "data"
LOGS_DIR          = BASE_DIR / "logs"
KNOWLEDGE_JSON    = DATA_DIR / "knowledge.json"
LOG_FILE          = LOGS_DIR / "chat_log.json"
CHAT_HISTORY_CSV  = LOGS_DIR / "chat_history_log.csv"   # ③ PDCA用ログ
LAST_SYNC_FILE    = DATA_DIR / "last_sync.txt"          # 施策同期完了日時

# RAG設定
RAG_TOP_POLICY  = 20   # 施策：スコア上位件数
RAG_TOP_MANUAL  = 30   # マニュアル：スコア上位件数
RAG_TOP_PRICING = 10   # 料金表：スコア上位件数

st.set_page_config(
    page_title="コールセンター AIアシスタント",
    page_icon="📞",
    layout="wide",
)

# ④ UIデザイン調整用CSS
st.markdown("""
<style>
/* ── 全体フォント ── */
html, body, [class*="css"] {
    font-size: 16px !important;
}

/* ── チャット吹き出し共通 ── */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
    line-height: 1.75;
    font-size: 15px !important;
}

/* ── ユーザー吹き出し（右寄り・青系） ── */
[data-testid="stChatMessage"][data-testid*="user"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #E8F4FD;
    border-left: 4px solid #2196F3;
}

/* ── AI吹き出し（左寄り・緑系） ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: #F1F8F4;
    border-left: 4px solid #4CAF50;
}

/* ── 入力ボックス ── */
[data-testid="stChatInput"] textarea {
    font-size: 15px !important;
    border-radius: 10px !important;
    border: 2px solid #2196F3 !important;
}

/* ── 入力ボックスのラベル ── */
[data-testid="stChatInput"] label {
    font-size: 14px !important;
    color: #555 !important;
}

/* ── タイトル ── */
h1 {
    color: #1A237E !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* ── サブヘッダー ── */
h2, h3 {
    color: #1565C0 !important;
}

/* ── サイドバー ── */
section[data-testid="stSidebar"] {
    background-color: #F5F7FA;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 15px !important;
}

/* ── フィードバックボタン ── */
button[kind="secondary"] {
    border-radius: 8px !important;
    font-size: 14px !important;
}

/* ── 知識ベース更新日時キャプション ── */
.stCaption {
    color: #888 !important;
    font-size: 12px !important;
}

/* ── スピナー文字 ── */
.stSpinner > div {
    font-size: 14px !important;
    color: #1565C0 !important;
}

/* ── divider ── */
hr {
    border-color: #E0E0E0 !important;
}
</style>
""", unsafe_allow_html=True)


# ---------- 知識ベース ----------

@st.cache_data(ttl=int(os.getenv("CACHE_TTL", 3600)))
def load_all_records() -> tuple[dict, str]:
    """knowledge.json を読み込み ({"policy":[], "manual":[], "pricing":[]}, fetched_at) を返す。"""
    if not KNOWLEDGE_JSON.exists():
        return {}, ""
    with open(KNOWLEDGE_JSON, encoding="utf-8") as f:
        data = json.load(f)
    fetched_at = data.get("fetched_at", "不明")
    records = {
        "policy":  data.get("policy",  []),
        "manual":  data.get("manual",  []),
        "pricing": data.get("pricing", []),
    }
    return records, fetched_at


def _tokenize(text: str) -> set:
    """テキストを簡易トークン集合に変換する（外部ライブラリ不要）。"""
    tokens = re.findall(r'[A-Za-z0-9０-９Ａ-Ｚａ-ｚ\u3040-\u9fff\u30a0-\u30ff]+', text)
    result = set()
    for t in tokens:
        if len(t) >= 2:
            result.add(t.lower())
            for i in range(len(t) - 1):
                result.add(t[i:i+2].lower())
    return result


def _clean_summary(text: str) -> str:
    """Notionのリッチテキスト記法（callout等）をClaudeに渡す前に除去する。"""
    text = re.sub(r'^:::[ \t]+callout[^\n]*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'^:::[ \t]*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _score(record: dict, query_tokens: set) -> int:
    """レコードと質問トークンのマッチ数を返す。"""
    target = (record.get("title", "") + " " + record.get("summary", "")).lower()
    return len(query_tokens & _tokenize(target))


def _expand_query(question: str) -> str:
    """Claudeを使って質問文を同義語・関連語で拡張し、検索用クエリ文字列を返す。"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return question
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=200,
            system=(
                "あなたはコールセンター向け社内マニュアル検索の専門家です。\n"
                "ユーザーの質問文を受け取り、検索ヒット率を上げるための"
                "同義語・関連語・上位概念語を追加した拡張クエリを返してください。\n"
                "出力形式：元の質問に続けて関連語をスペース区切りで追記した1行のテキストのみ。\n"
                "例）「返品の手続きを教えて」→「返品の手続きを教えて 返品 交換 送り返す 返送 キャンセル 手順 方法\"\n"
                "余計な説明・記号・改行は不要。"
            ),
            messages=[{"role": "user", "content": question}],
        )
        expanded = resp.content[0].text.strip()
        return expanded if expanded else question
    except Exception:
        return question


def retrieve_knowledge(question: str, records: dict) -> tuple[str, list[str]]:
    """
    質問文に関連するレコードをRAGで絞り込み、
    (Claudeに渡すテキスト, 参照したサマリーIDリスト) を返す。
    """
    expanded_query = _expand_query(question)
    query_tokens = _tokenize(expanded_query)

    section_meta = [
        ("policy",  "【施策】（最優先：特例・キャンペーン情報）",  RAG_TOP_POLICY),
        ("manual",  "【マニュアル】（ルール・手順）",             RAG_TOP_MANUAL),
        ("pricing", "【料金表】（価格・プラン情報）",             RAG_TOP_PRICING),
    ]

    sections = []
    retrieved_counts = {}
    ref_ids: list[str] = []   # ③ 参照サマリーIDを収集

    for key, header, top_n in section_meta:
        all_recs = records.get(key, [])
        if not all_recs:
            continue

        scored = sorted(all_recs, key=lambda r: _score(r, query_tokens), reverse=True)
        selected = scored[:top_n]

        block = [f"## {header}"]
        for r in selected:
            block.append(f"### {r['title']}\n{_clean_summary(r['summary'])}")
            if r.get("id"):
                ref_ids.append(str(r["id"]))
        sections.append("\n".join(block))
        retrieved_counts[key] = len(selected)

    body = "\n\n".join(sections)
    summary_line = (
        f"[RAG検索結果: 施策{retrieved_counts.get('policy',0)}件 / "
        f"マニュアル{retrieved_counts.get('manual',0)}件 / "
        f"料金表{retrieved_counts.get('pricing',0)}件 / "
        f"拡張クエリ: {expanded_query[:60]}{'…' if len(expanded_query) > 60 else ''}]"
    )
    knowledge_text = f"{summary_line}\n優先順位: 施策 > マニュアル > 料金表\n\n{body}"
    return knowledge_text, ref_ids


# ① 回答の厳格化 + ② 金額フォーマット を組み込んだシステムプロンプト
SYSTEM_PROMPT_TEMPLATE = """\
あなたは通販コールセンターのAIアシスタントです。
パートスタッフの質問に、正確・簡潔・丁寧に回答してください。

## 回答ルール

### ① 根拠の厳格化（推測厳禁）
- 「〜だと思われます」「〜かもしれません」「おそらく〜」「〜と考えられます」など、
  推測・曖昧な表現は**一切使用禁止**。
- 知識ベースに記載のない事項については、必ず以下の文言で回答すること：
  > 「該当する情報が知識ベースに記載がありません。直近のチャットワーク等の共有事項を確認するか、SV（スーパーバイザー）へ確認してください。」
- 部分的にしか情報がない場合も、わかる範囲だけを明記し、不明部分は上記フレーズで案内する。

### ② 金額案内フォーマットの厳守
- 金額を回答する際は**必ず**以下のいずれかの形式を使用すること：
  - `税込〇〇円（送料別）`
  - `税込〇〇円（送料込）`
- 送料の記載がない場合も「（送料別）」または「（送料込）」のどちらかを必ず明記すること。
- 「円」だけで終わらせない。税込・送料の扱いを必ずセットで記載する。

### ③ 優先順位
1. 知識ベースを必ず参照し、「施策 → マニュアル → 料金表」の優先順位で回答する。
2. マニュアルと施策で情報が矛盾する場合は、まずマニュアルの標準ルールを回答し、
   「ただし、現在〇〇の施策が適用されている場合は例外となります」と添える。

### ④ 回答スタイル
- 箇条書きを活用し、要点を先に述べる。
- 結論を最初に一言で示してから、詳細を続ける。

## 知識ベース
{knowledge}
"""


# ---------- ③ PDCA用CSVログ ----------

CSV_FIELDNAMES = ["timestamp", "question", "answer", "ref_ids", "feedback"]


def append_chat_history_csv(
    question: str,
    answer: str,
    ref_ids: list[str],
    feedback: str = "",
) -> None:
    """chat_history_log.csv に1行追記する。"""
    LOGS_DIR.mkdir(exist_ok=True)
    is_new = not CHAT_HISTORY_CSV.exists()
    with open(CHAT_HISTORY_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question":  question,
            "answer":    answer,
            "ref_ids":   "|".join(ref_ids),   # 複数IDを | 区切りで格納
            "feedback":  feedback,
        })


def update_csv_feedback(question: str, timestamp: str, feedback: str) -> None:
    """chat_history_log.csv の該当行の feedback 列を更新する。"""
    if not CHAT_HISTORY_CSV.exists():
        return
    rows = []
    with open(CHAT_HISTORY_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["timestamp"] == timestamp and row["question"] == question:
                row["feedback"] = feedback
            rows.append(row)
    with open(CHAT_HISTORY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# ---------- 施策即時同期 ----------

def read_last_sync() -> str:
    """last_sync.txt から最終同期日時文字列を返す。ファイルがなければ空文字。"""
    if not LAST_SYNC_FILE.exists():
        return ""
    try:
        line = LAST_SYNC_FILE.read_text(encoding="utf-8").strip()
        # フォーマット: "2026-03-05 14:30:00\t施策のみ"
        parts = line.split("\t")
        dt_str = parts[0]
        scope  = parts[1] if len(parts) > 1 else ""
        return f"{dt_str}（{scope}）" if scope else dt_str
    except Exception:
        return ""


def run_policy_sync() -> tuple[bool, str]:
    """
    fetch_knowledge.py --policy-only のみを実行する（10秒以内完了）。
    ※ AIサマリー生成（auto_summarize.py）は毎朝の夜間バッチ専用。
      このボタンは「Notionに書かれた最新内容をそのまま即取得」する用途。
    """
    python = sys.executable

    try:
        # Notionから施策DBを取得して knowledge.json の policy 部分だけ差し替え
        r = subprocess.run(
            [python, str(BASE_DIR / "fetch_knowledge.py"), "--policy-only"],
            capture_output=True, text=True, timeout=60,   # 60秒で十分
            cwd=str(BASE_DIR),
        )
        if r.returncode != 0:
            return False, f"取得エラー:\n{r.stderr[-400:]}"

        # Streamlit のキャッシュをクリアして最新 knowledge.json を読み直す
        load_all_records.clear()

        last_sync = read_last_sync()
        return True, last_sync

    except subprocess.TimeoutExpired:
        return False, "タイムアウト（60秒超過）。ネットワークを確認してください。"
    except Exception as e:
        return False, str(e)


# ---------- ログ（既存 JSON ログ） ----------

def load_log() -> list:
    LOGS_DIR.mkdir(exist_ok=True)
    if LOG_FILE.exists():
        with open(LOG_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_log(log: list):
    LOGS_DIR.mkdir(exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


# ---------- Claude API ----------

def get_answer(question: str, knowledge: str, history: list = None) -> str:
    """
    history: [{"role": "user"|"assistant", "content": str}, ...] の過去の会話履歴。
    Noneの場合は単発質問として扱う。
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "[ERROR] ANTHROPIC_API_KEY が未設定です"
    client = anthropic.Anthropic(api_key=api_key)
    system = SYSTEM_PROMPT_TEMPLATE.format(knowledge=knowledge)
    messages = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})
    resp = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    return resp.content[0].text


# ---------- ページ: チャット ----------

def _get_api_history() -> list:
    """Claude API に渡すための {"role", "content"} リストを返す。"""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]


def _submit_question(question: str, records: dict, use_history: bool = False):
    """質問を送信してログに記録する共通処理。"""
    history = _get_api_history() if use_history else None
    knowledge, ref_ids = retrieve_knowledge(question, records)   # ③ ref_ids を受け取る

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🔍 知識ベースを検索して回答を生成中..."):
            answer = get_answer(question, knowledge, history)
        st.markdown(answer)

    entry_id  = str(uuid.uuid4())
    timestamp = datetime.now().isoformat(timespec="seconds")

    st.session_state.messages.append({
        "role":         "assistant",
        "content":      answer,
        "id":           entry_id,
        "timestamp":    timestamp,
        "question":     question,
        "ref_ids":      ref_ids,
        "show_feedback": True,
    })

    # 既存 JSON ログ
    entry = {
        "id":        entry_id,
        "timestamp": timestamp,
        "question":  question,
        "answer":    answer,
        "feedback":  None,
    }
    st.session_state.log.append(entry)
    save_log(st.session_state.log)

    # ③ PDCA用 CSV ログに追記
    append_chat_history_csv(question, answer, ref_ids)


def page_chat():
    st.title("📞 コールセンター AIアシスタント")

    # サブタイトル
    st.markdown(
        "<p style='color:#555; font-size:14px; margin-top:-12px;'>"
        "パートスタッフ向け質問サポート｜知識ベースを参照して正確に回答します"
        "</p>",
        unsafe_allow_html=True,
    )

    records, fetched_at = load_all_records()
    if not records:
        st.error("⚠️ 知識ファイルがありません。fetch_knowledge.py を実行してください。")
        return
    if fetched_at:
        st.caption(f"📚 知識ベース最終更新: {fetched_at}")

    st.divider()

    # セッション初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "log" not in st.session_state:
        st.session_state.log = load_log()
    if "deepdive_id" not in st.session_state:
        st.session_state.deepdive_id = None

    # ウェルカムメッセージ（会話がまだない場合）
    if not st.session_state.messages:
        st.info(
            "💡 **使い方**\n\n"
            "下の入力欄に質問を入力して送信してください。\n"
            "返品・解約・キャンペーン・料金など、業務に関わることをなんでも聞けます。\n\n"
            "⚠️ 知識ベースに記載がない場合は「記載がありません」とお伝えし、SVへの確認を促します。"
        )

    # 過去の会話を表示
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("show_feedback"):
            _render_feedback(msg["id"])

    # 深掘り入力欄（最新の回答に対してのみ表示）
    if st.session_state.deepdive_id:
        st.divider()
        st.caption("💬 前の回答をふまえて、さらに質問できます")
        deepdive_q = st.chat_input(
            "深掘り質問を入力してください...",
            key="deepdive_input",
        )
        if deepdive_q:
            st.session_state.deepdive_id = None
            _submit_question(deepdive_q, records, use_history=True)
            st.rerun()

    # 通常入力欄
    else:
        question = st.chat_input("質問を入力してください（例：返品の手続きを教えて）...")
        if question:
            _submit_question(question, records, use_history=False)
            st.rerun()


def _render_feedback(entry_id: str):
    col1, col2, col3, col4 = st.columns([2, 2, 3, 3])
    with col1:
        if st.button("✅ 解決", key=f"ok_{entry_id}", use_container_width=True):
            _update_feedback(entry_id, "resolved")
            st.success("フィードバックを記録しました")
    with col2:
        if st.button("❌ 未解決", key=f"ng_{entry_id}", use_container_width=True):
            _update_feedback(entry_id, "unresolved")
            st.warning("フィードバックを記録しました")
    with col3:
        if st.button("🔍 さらに深ぼる", key=f"deep_{entry_id}", use_container_width=True):
            st.session_state.deepdive_id = entry_id
            st.rerun()
    with col4:
        if st.button("💬 別の質問をする", key=f"new_{entry_id}", use_container_width=True):
            st.session_state.messages = []
            st.session_state.deepdive_id = None
            st.rerun()


def _update_feedback(entry_id: str, value: str):
    log = st.session_state.log
    for entry in log:
        if entry["id"] == entry_id:
            entry["feedback"] = value
            break
    save_log(log)

    # ③ CSV ログのフィードバックも更新
    for msg in st.session_state.messages:
        if msg.get("id") == entry_id:
            update_csv_feedback(
                question=msg.get("question", ""),
                timestamp=msg.get("timestamp", ""),
                feedback=value,
            )
            msg["show_feedback"] = False
            break
    else:
        # session_state に question/timestamp がない場合はフォールバック
        for msg in st.session_state.messages:
            if msg.get("id") == entry_id:
                msg["show_feedback"] = False


# ---------- ページ: 管理画面 ----------

def page_admin():
    st.title("📊 管理画面")

    log = load_log()
    if not log:
        st.info("ログがまだありません。チャット画面で質問してみましょう。")
        return

    # --- サマリー ---
    total      = len(log)
    resolved   = sum(1 for e in log if e.get("feedback") == "resolved")
    unresolved = sum(1 for e in log if e.get("feedback") == "unresolved")
    no_fb      = total - resolved - unresolved

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("総質問数", total)
    c2.metric("✅ 解決", resolved)
    c3.metric("❌ 未解決", unresolved)
    c4.metric("フィードバックなし", no_fb)

    st.divider()

    # --- 直近30日の日別質問数 ---
    st.subheader("日別質問数（直近30日）")
    now = datetime.now()
    counts: Counter = Counter()
    for e in log:
        try:
            dt = datetime.fromisoformat(e["timestamp"])
            if now - dt <= timedelta(days=30):
                counts[dt.strftime("%m/%d")] += 1
        except Exception:
            pass
    if counts:
        import pandas as pd
        df_daily = pd.DataFrame(
            sorted(counts.items()), columns=["日付", "質問数"]
        ).set_index("日付")
        st.bar_chart(df_daily)

    # --- 未解決ログ一覧 ---
    st.subheader("未解決の質問一覧")
    unresolved_list = [e for e in log if e.get("feedback") == "unresolved"]
    if unresolved_list:
        for e in sorted(unresolved_list, key=lambda x: x["timestamp"], reverse=True):
            with st.expander(f"[{e['timestamp']}] {e['question'][:60]}"):
                st.markdown(f"**質問:** {e['question']}")
                st.markdown(f"**回答:** {e['answer']}")
    else:
        st.success("未解決の質問はありません")

    # --- PDCA用CSV ダウンロード ---
    st.divider()
    st.subheader("📥 PDCA用ログ（chat_history_log.csv）")
    if CHAT_HISTORY_CSV.exists():
        with open(CHAT_HISTORY_CSV, encoding="utf-8-sig") as f:
            csv_bytes = f.read().encode("utf-8-sig")
        st.download_button(
            label="📥 chat_history_log.csv をダウンロード",
            data=csv_bytes,
            file_name="chat_history_log.csv",
            mime="text/csv",
        )
        # プレビュー（最新10件）
        import pandas as pd
        df_csv = pd.read_csv(CHAT_HISTORY_CSV, encoding="utf-8-sig")
        st.dataframe(
            df_csv.tail(10)[["timestamp", "question", "feedback"]].rename(columns={
                "timestamp": "日時",
                "question": "質問",
                "feedback": "フィードバック",
            }),
            use_container_width=True,
        )
    else:
        st.info("まだ chat_history_log.csv がありません。チャット画面で質問すると自動作成されます。")

    # --- 全ログ ---
    st.divider()
    st.subheader("全ログ（最新50件）")
    import pandas as pd
    rows = []
    for e in sorted(log, key=lambda x: x["timestamp"], reverse=True)[:50]:
        rows.append({
            "日時":          e["timestamp"],
            "質問":          e["question"][:60],
            "フィードバック": e.get("feedback") or "なし",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---------- ページ: Notion編集 ----------

def page_notion_edit():
    st.title("✏️ Notion 編集")

    notion_key = os.getenv("NOTION_API_KEY")
    if not notion_key:
        st.error("NOTION_API_KEY が未設定です")
        return

    db_map = {
        "施策":      os.getenv("NOTION_DB_POLICY"),
        "マニュアル": os.getenv("NOTION_DB_MANUAL"),
        "料金表":    os.getenv("NOTION_DB_PRICING"),
    }

    db_name = st.selectbox("編集するDB", list(db_map.keys()))
    db_id = db_map[db_name]
    if not db_id:
        st.warning(f"{db_name} の DB ID が .env に設定されていません")
        return

    client = NotionClient(auth=notion_key)

    @st.cache_data(ttl=60, show_spinner=False)
    def fetch_pages(did: str):
        resp = client.databases.query(database_id=did, page_size=50)
        pages = []
        for p in resp["results"]:
            props = p["properties"]
            title = ""
            for key in ["名前", "Name", "タイトル", "Title"]:
                if key in props:
                    title = "".join(t["plain_text"] for t in props[key].get("title", []))
                    break
            if not title:
                for val in props.values():
                    if val.get("type") == "title":
                        title = "".join(t["plain_text"] for t in val.get("title", []))
                        break
            summary = ""
            if "サマリー" in props:
                prop = props["サマリー"]
                if prop.get("type") == "rich_text":
                    summary = "".join(t["plain_text"] for t in prop.get("rich_text", []))
            pages.append({"id": p["id"], "title": title or "（無題）", "summary": summary})
        return pages

    pages = fetch_pages(db_id)
    if not pages:
        st.info("ページが見つかりません")
        return

    titles = [p["title"] for p in pages]
    selected = st.selectbox("ページを選択", titles)
    page = next(p for p in pages if p["title"] == selected)

    st.markdown(f"**ページID:** `{page['id']}`")
    new_summary = st.text_area("サマリーを編集", value=page["summary"], height=200)

    if st.button("Notionに保存"):
        try:
            client.pages.update(
                page_id=page["id"],
                properties={
                    "サマリー": {
                        "rich_text": [{"text": {"content": new_summary}}]
                    }
                },
            )
            st.success("保存しました。fetch_knowledge.py を再実行して知識ファイルを更新してください。")
            fetch_pages.clear()
        except Exception as ex:
            st.error(f"保存に失敗しました: {ex}")


# ---------- メイン ----------

def main():
    with st.sidebar:
        st.markdown(
            "<h2 style='font-size:18px; color:#1A237E; margin-bottom:4px;'>📞 AIアシスタント</h2>",
            unsafe_allow_html=True,
        )
        st.caption("コールセンター業務サポートツール")
        st.divider()
        page = st.radio(
            "ページ",
            ["💬 チャット", "📊 管理画面", "✏️ Notion編集"],
            label_visibility="collapsed",
        )
        st.divider()

        # ── ⚡ 施策即時同期ボタン ──────────────────────────
        st.markdown(
            "<p style='font-size:12px; color:#555; margin-bottom:4px;'>"
            "⚡ <b>施策を今すぐ同期</b></p>",
            unsafe_allow_html=True,
        )
        st.caption("Notionの施策DBを最新化してAIに即反映します")

        # セッション初期化
        if "sync_status" not in st.session_state:
            st.session_state.sync_status = None   # None / "running" / "ok" / "error"
        if "sync_message" not in st.session_state:
            st.session_state.sync_message = ""

        # 最終同期日時の表示
        last_sync = read_last_sync()
        if last_sync:
            st.caption(f"最終同期: {last_sync}")

        sync_btn = st.button(
            "🔄 最新の施策情報を同期",
            key="sync_policy_btn",
            use_container_width=True,
        )
        if sync_btn:
            st.session_state.sync_status  = "running"
            st.session_state.sync_message = ""

        if st.session_state.sync_status == "running":
            with st.spinner("⚡ 施策DBを同期中... (1〜2分かかります)"):
                ok, msg = run_policy_sync()
            if ok:
                st.session_state.sync_status  = "ok"
                st.session_state.sync_message = msg
            else:
                st.session_state.sync_status  = "error"
                st.session_state.sync_message = msg
            st.rerun()

        if st.session_state.sync_status == "ok":
            st.success(f"✅ 同期完了\n最終更新: {st.session_state.sync_message}")
        elif st.session_state.sync_status == "error":
            st.error(f"❌ 同期失敗\n{st.session_state.sync_message}")

        st.divider()

        # ── 知識ベースキャッシュクリア ───────────────────────
        if st.button("🗂 キャッシュのみクリア", use_container_width=True,
                     help="Notionへの再取得はせず、メモリキャッシュだけクリアします"):
            load_all_records.clear()
            st.session_state.sync_status = None
            st.success("キャッシュをクリアしました")

        st.markdown(
            "<p style='font-size:11px; color:#aaa; margin-top:20px;'>"
            "© Q&Ai コールセンター支援ツール</p>",
            unsafe_allow_html=True,
        )

    # ラベルからアイコンを除いてページ判定
    if "チャット" in page:
        page_chat()
    elif "管理画面" in page:
        page_admin()
    else:
        page_notion_edit()


if __name__ == "__main__":
    main()
