"""
app.py
Streamlit製チャット＆管理画面。

実行方法:
  streamlit run app.py
"""

import concurrent.futures
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
from google import genai as google_genai
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

# RAG設定（件数を絞ることでトークン数を削減し応答速度を向上）
RAG_TOP_POLICY   = 10   # 施策：スコア上位件数
RAG_TOP_MANUAL   = 15   # マニュアル：スコア上位件数
RAG_TOP_PRICING  =  5   # 料金表：スコア上位件数
RAG_TOP_PRODUCTS =  5   # 商品詳細：スコア上位件数

# ---------- Gemini 音声認識 ----------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def transcribe_audio_gemini(audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    """Gemini APIで音声をテキストに変換する"""
    import base64
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY が設定されていません")
    client = google_genai.Client(api_key=GEMINI_API_KEY)
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": audio_b64,
                        }
                    },
                    {
                        "text": (
                            "あなたはコールセンターのオペレーター向け音声認識システムです。"
                            "以下の音声を正確に日本語テキストに書き起こしてください。\n\n"
                            "【重要】この音声はコールセンター業務に関する質問です。"
                            "よく出てくる言葉: 返品、解約、定期便、キャンセル、料金、コース変更、"
                            "スラヘル、オモニストン、シロッシュ、デイリーワン、するるん緑茶、"
                            "サラフィネ、ばぶりーキッズ、引き上げ、掘り起こし、アウトバウンド。\n\n"
                            "書き起こし結果のテキストのみを出力してください。説明や前置きは不要です。"
                        )
                    },
                ]
            }
        ],
    )
    return response.text.strip()

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
        "policy":   data.get("policy",   []),
        "manual":   data.get("manual",   []),
        "pricing":  data.get("pricing",  []),
        "products": data.get("products", []),
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
                "あなたは通販コールセンター向け社内マニュアル検索の専門家です。\n"
                "ユーザーの質問文を受け取り、検索ヒット率を上げるための"
                "同義語・関連語・上位概念語を追加した拡張クエリを返してください。\n"
                "出力形式：元の質問に続けて関連語をスペース区切りで追記した1行のテキストのみ。\n"
                "余計な説明・記号・改行は不要。\n"
                "\n"
                "【変換例：商品・成分系】\n"
                "「返品の手続きを教えて」→「返品の手続きを教えて 返品 交換 送り返す 返送 キャンセル 手順 方法\"\n"
                "「スラヘルの成分を教えて」→「スラヘルの成分を教えて SLAHEL 原材料 原材料名 有効成分 機能性成分 配合 成分表\"\n"
                "「この商品の特徴は」→「この商品の特徴は 訴求ポイント 商品特徴 こだわり 効果 機能 説明\"\n"
                "\n"
                "【変換例：体調・副作用系】\n"
                "「飲んだら気持ち悪くなった」→「飲んだら気持ち悪くなった 副作用 体調不良 吐き気 腹痛 頭痛 好転反応 使用中止\"\n"
                "「お腹がゆるくなった」→「お腹がゆるくなった 下痢 軟便 腸内 お腹を壊す 副作用 体調不良 摂取量\"\n"
                "「歯が染みる」→「歯が染みる 知覚過敏 副作用 使用中止 刺激 ピリピリ\"\n"
                "「ピリピリする」→「ピリピリする 刺激 知覚過敏 副作用 体調不良 使用方法\"\n"
                "\n"
                "【変換例：効果・継続系】\n"
                "「全然変わらない」→「全然変わらない 効果なし 変化がない 即効性 継続 個人差 目安期間 3ヶ月\"\n"
                "「どのくらい続ければいい」→「どのくらい続ければいい 継続期間 目安 3ヶ月 効果 使用期間\"\n"
                "「飲み忘れたらどうする」→「飲み忘れたらどうする 飲み忘れ 継続 まとめ飲み 服用 摂取\"\n"
                "\n"
                "【変換例：使い方・トラブル系】\n"
                "「使いにくい」→「使いにくい 使い方 コツ 使用方法 注意 改善 こぼれる 扱い方\"\n"
                "「固まってしまった」→「固まってしまった 塊 品質 保管 湿気 異物 問題 使用可否\"\n"
                "「白い塊がある」→「白い塊がある 結晶 品質 異物 カビ 使用可否 対処法\"\n"
                "\n"
                "【変換例：不安・安全系】\n"
                "「大丈夫ですか」→「大丈夫ですか 安全性 副作用 使用可否 成分 アレルギー 注意事項\"\n"
                "「心配なんですが」→「心配なんですが 安全性 副作用 成分 アレルギー 使用上の注意\"\n"
                "「病気に効きますか」→「病気に効きますか 効果効能 薬 医薬品 機能性表示食品 食品 治療\"\n"
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
        ("policy",   "【施策】（最優先：特例・キャンペーン情報）",  RAG_TOP_POLICY),
        ("manual",   "【マニュアル】（ルール・手順）",             RAG_TOP_MANUAL),
        ("pricing",  "【料金表】（価格・プラン情報）",             RAG_TOP_PRICING),
        ("products", "【商品詳細】（商品名・価格・カテゴリ）",     RAG_TOP_PRODUCTS),
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
        f"商品詳細{retrieved_counts.get('products',0)}件 / "
        f"拡張クエリ: {expanded_query[:60]}{'…' if len(expanded_query) > 60 else ''}]"
    )
    knowledge_text = f"{summary_line}\n優先順位: 施策 > マニュアル > 料金表 > 商品詳細\n\n{body}"
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
1. 知識ベースを必ず参照し、「施策 → マニュアル → 料金表 → 商品詳細」の優先順位で回答する。
2. 商品の成分・原材料・特徴・こだわり・使い方・QAに関する質問は【商品詳細】セクションを必ず参照する。
3. マニュアルと施策で情報が矛盾する場合は、まずマニュアルの標準ルールを回答し、
   「ただし、現在〇〇の施策が適用されている場合は例外となります」と添える。

### ④ 回答スタイル
- 箇条書きを活用し、要点を先に述べる。
- 結論を最初に一言で示してから、詳細を続ける。

### ⑤ 商品案内の共通ルール（全商品対応）

商品に関する質問は、知識ベースの【商品詳細】セクションに記載された情報を**そのまま正確に**使って回答すること。
以下のルールを必ず守る。

**【成分・原材料】**
- 「成分は？」「何が入ってる？」→ 原材料名・ホワイトニング成分・有効成分など、記載されている成分情報を全て案内する。
- 「〇〇は入っていますか？」→ 原材料一覧を確認し、入っていれば「含まれています」、なければ「含まれていません」と明確に答える。推測で答えない。

**【使い方・使用方法】**
- 「どうやって使うの？」→ 商品詳細の使用方法・使用目安・タイミングを案内する。
- 使い方に注意事項（乾いた状態で使う、等）がある場合は必ずセットで伝える。

**【効果・効能】**
- 「どんな効果がある？」→ 訴求ポイント・商品特徴・機能性成分の説明を使って案内する。
- 「薬ですか？」「病気に効きますか？」→ 機能性表示食品・化粧品・食品など、商品のカテゴリを明確に伝えた上で「病気の治療・治癒を目的とするものではありません」と案内する。

**【アレルギー】**
- 「アレルギーがあるけど使えますか？」→ まず「アレルギーがある場合は、原材料を確認の上ご使用ください」と伝え、知識ベースにアレルギー情報の記載があればそれを案内する。記載がなければSVへ確認を促す。

**【外観・品質トラブル】**
- 「変な色・塊・結晶がある」「カビが生えた」→ 必ず「色・状態」を確認してから対応を案内する。知識ベースに対応フローが記載されている場合はその通りに案内する。

**【他の商品・薬との併用】**
- 「他の商品と一緒に使えますか？」「薬を飲んでいるけど大丈夫？」→ 知識ベースに記載がある場合はそれを案内する。記載がない場合は「医師・薬剤師にご相談ください」と案内する。

**【即効性・効果が出ない】**
- 「全然変わらない」「効かない」→ 知識ベースに継続期間の目安が記載されている場合はそれを案内し、個人差があることを伝える。

**【妊娠・授乳中・子供の使用】**
- 「妊娠中でも大丈夫？」「授乳中は？」「子供は使える？何歳から？」→ 商品ごとに対応が異なるため、必ず知識ベースの該当商品の記載を確認して案内する。
  - 「推奨しない」と記載がある場合→「推奨しておりません。担当の医師にご相談ください」と案内する。
  - 「問題ない」と記載がある場合→「特に問題ありません。ご心配な場合はかかりつけ医にご相談ください」と案内する。
  - 記載がない場合→「念のため担当の医師にご相談の上ご使用ください」と案内する。
  - 年齢制限の記載がある場合は必ずその年齢を伝える。

**【摂取量・飲み過ぎ・上限】**
- 「多めに飲んでいい？」「上限は？」「飲み過ぎるとどうなる？」→ 知識ベースに上限・注意事項の記載がある場合は必ずそれを案内する。
  - 上限が明記されている商品（海宝の力など）は「1日〇〇が上限です」と数値を明示する。
  - 「過剰摂取は避けてください」と記載がある場合はその旨を伝える。
  - 記載がない場合は「1日の摂取目安量を守ってご使用ください」と案内する。

**【解約を検討しているお客様への継続応援】**
- 「効かないから解約したい」「味が嫌い」「使いにくい」などの解約理由が含まれる質問→ 知識ベースに継続応援トークや切り返しが記載されている場合は、その内容を使って案内する。
  - 効果実感なし→ 継続期間の目安・個人差・気づきにくい変化を案内する。
  - 使いにくさ→ 使い方のコツ・代替の使い方を案内する。
  - 商品によって「継続応援しない（卒業制度など）」と明記がある場合はその通りに対応する。

### ⑥ ニュアンスの言い換え解釈（口語→正式用語への変換）

パートスタッフやお客様は口語・感覚的な言葉で質問してくることが多い。
以下の対応表を参考に、言葉の意図を正しく読み取って知識ベースを参照すること。

**【体調・副作用系の言い換え】**
| 言われた言葉 | 知識ベースで検索すべき概念 |
|------------|------------------------|
| 「気持ち悪くなった」「吐き気がする」 | 副作用・体調不良・好転反応・使用中止 |
| 「お腹がゆるい」「お腹を壊した」 | 下痢・軟便・腸内・摂取量を減らす |
| 「歯が染みる」「ピリピリする」 | 知覚過敏・刺激・副作用・使用中止 |
| 「頭が痛くなった」 | 副作用・体調不良・服用中止・医師相談 |
| 「体がかゆい」「ブツブツが出た」 | アレルギー・副作用・使用中止・医師相談 |

**【効果・継続系の言い換え】**
| 言われた言葉 | 知識ベースで検索すべき概念 |
|------------|------------------------|
| 「全然変わらない」「効いてる気がしない」 | 変化がない・継続期間・個人差・3ヶ月 |
| 「どのくらいで効く？」「すぐ効く？」 | 継続期間・目安・即効性・個人差 |
| 「飲み忘れたらどうする？」 | 飲み忘れ・まとめ飲み禁止・継続・服用方法 |
| 「休んでもいい？」「毎日じゃないとダメ？」 | 継続・毎日・使用頻度・効果 |

**【使い方・トラブル系の言い換え】**
| 言われた言葉 | 知識ベースで検索すべき概念 |
|------------|------------------------|
| 「使いにくい」「扱いにくい」 | 使用方法・コツ・注意事項・使い方 |
| 「こぼれる」「粉が落ちる」 | 使用方法・コツ・保管・容器の扱い |
| 「固まってしまった」「塊がある」 | 品質・保管・結晶・湿気・使用可否 |
| 「カビが生えた」「変な色がついている」 | 品質トラブル・色・返送・代品手配 |
| 「臭いがおかしい」「味がおかしい」 | 品質・異常・保管・使用可否 |

**【安全・不安系の言い換え】**
| 言われた言葉 | 知識ベースで検索すべき概念 |
|------------|------------------------|
| 「大丈夫？」「心配なんですが」 | 安全性・副作用・成分・注意事項 |
| 「薬と一緒に飲んでいい？」 | 薬との併用・医師相談・飲み合わせ |
| 「病気に効く？」「治る？」 | 効果効能・機能性表示食品・食品・薬ではない |
| 「本当に効くの？」「詐欺じゃない？」 | 安全性・機能性・臨床試験・成分根拠 |

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

def get_answer(question: str, knowledge: str, history: list = None):
    """
    history: [{"role": "user"|"assistant", "content": str}, ...] の過去の会話履歴。
    Noneの場合は単発質問として扱う。
    ストリーミングでトークンを逐次 yield する。
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        yield "[ERROR] ANTHROPIC_API_KEY が未設定です"
        return
    client = anthropic.Anthropic(api_key=api_key)
    system = SYSTEM_PROMPT_TEMPLATE.format(knowledge=knowledge)
    messages = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})
    with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=512,
        system=system,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


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
        # ストリーミング表示：最初のトークンが来た瞬間から表示開始
        answer = st.write_stream(get_answer(question, knowledge, history))

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
    if "voice_text" not in st.session_state:
        st.session_state.voice_text = ""

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
        # 音声入力エリア（GeminiAPIキーがある場合のみ表示）
        if GEMINI_API_KEY:
            with st.expander("🎤 音声で質問する", expanded=False):
                st.caption("マイクボタンを押して話しかけてください。録音後に自動でテキスト変換します。")
                audio_input = st.audio_input("録音する", key="audio_recorder")
                if audio_input is not None:
                    with st.spinner("音声を変換中..."):
                        try:
                            audio_bytes = audio_input.read()
                            mime_type = getattr(audio_input, "type", None) or "audio/wav"
                            transcribed = transcribe_audio_gemini(audio_bytes, mime_type=mime_type)
                            st.session_state.voice_text = transcribed
                            st.rerun()
                        except Exception as e:
                            st.error(f"音声変換エラー: {e}")

            # 音声変換結果がある場合、テキスト編集欄＋送信ボタンを表示
            if st.session_state.voice_text:
                col_text, col_btn = st.columns([5, 1])
                with col_text:
                    edited_text = st.text_input(
                        "音声入力の内容（修正可能）",
                        value=st.session_state.voice_text,
                        key="voice_edit_input",
                        label_visibility="collapsed",
                    )
                with col_btn:
                    if st.button("送信", key="voice_send_btn", use_container_width=True, type="primary"):
                        q = edited_text
                        st.session_state.voice_text = ""
                        _submit_question(q, records, use_history=False)
                        st.rerun()

        question = st.chat_input("質問を入力してください（例：返品の手続きを教えて）...")
        if question:
            st.session_state.voice_text = ""
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

    # --- 週次レポート送信 ---
    st.divider()
    st.subheader("📤 週次品質レポートをSlackに送信")
    st.caption("直近7日分のログを集計し、AI分析付きPDFをSlackに送信します")

    col_days, col_btn, col_dry = st.columns([2, 2, 2])
    with col_days:
        report_days = st.selectbox("集計期間", [7, 14, 30], index=0, key="report_days",
                                   format_func=lambda x: f"直近{x}日")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        send_btn = st.button("📤 今すぐSlackに送信", use_container_width=True, type="primary",
                             key="send_report_btn")
    with col_dry:
        st.markdown("<br>", unsafe_allow_html=True)
        dry_btn = st.button("📄 PDFのみ生成（送信なし）", use_container_width=True,
                            key="dry_report_btn")

    if send_btn or dry_btn:
        is_dry = dry_btn
        with st.spinner("レポート生成中..."):
            try:
                import importlib.util, sys as _sys
                spec = importlib.util.spec_from_file_location(
                    "weekly_report", BASE_DIR / "weekly_report.py"
                )
                wr = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(wr)
                ok, msg = wr.run(days=report_days, dry_run=is_dry)
            except Exception as ex:
                ok, msg = False, str(ex)

        if ok:
            st.success(f"✅ {msg}")
            # PDFダウンロードボタンを表示
            report_dir = BASE_DIR / "logs" / "reports"
            pdfs = sorted(report_dir.glob("*.pdf"), reverse=True) if report_dir.exists() else []
            if pdfs:
                latest_pdf = pdfs[0]
                st.download_button(
                    label=f"📥 {latest_pdf.name} をダウンロード",
                    data=latest_pdf.read_bytes(),
                    file_name=latest_pdf.name,
                    mime="application/pdf",
                )
        else:
            st.error(f"❌ {msg}")


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

ADMIN_PASSWORD = "kaede1125"


def main():
    with st.sidebar:
        st.markdown(
            "<h2 style='font-size:18px; color:#1A237E; margin-bottom:4px;'>📞 AIアシスタント</h2>",
            unsafe_allow_html=True,
        )
        st.caption("別途業務サポートツール")
        st.divider()

        # ── ページ選択（ブロック形式）──────────────────────────
        st.markdown(
            "<p style='font-size:12px; color:#888; margin-bottom:8px;'>ページを選択</p>",
            unsafe_allow_html=True,
        )

        if "page" not in st.session_state:
            st.session_state.page = "chat"

        if st.button("💬  チャット", use_container_width=True,
                     type="primary" if st.session_state.page == "chat" else "secondary"):
            st.session_state.page = "chat"
            st.rerun()
        if st.button("📊  管理画面", use_container_width=True,
                     type="primary" if st.session_state.page in ("admin", "admin_login") else "secondary"):
            st.session_state.page = "admin_login"
            st.rerun()
        if st.button("✏️  概念編集", use_container_width=True,
                     type="primary" if st.session_state.page == "notion" else "secondary"):
            st.session_state.page = "notion"
            st.rerun()

        page = st.session_state.page
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

    # ページ判定
    if page == "chat":
        page_chat()
    elif page == "admin_login":
        # パスワード認証
        if "admin_authenticated" not in st.session_state:
            st.session_state.admin_authenticated = False

        if st.session_state.admin_authenticated:
            st.session_state.page = "admin"
            page_admin()
        else:
            st.title("🔐 管理画面")
            st.markdown("管理画面にアクセスするにはパスワードを入力してください。")
            pw = st.text_input("パスワード", type="password", key="admin_pw_input")
            if st.button("ログイン", type="primary"):
                if pw == ADMIN_PASSWORD:
                    st.session_state.admin_authenticated = True
                    st.session_state.page = "admin"
                    st.rerun()
                else:
                    st.error("パスワードが違います")
    elif page == "admin":
        if st.session_state.get("admin_authenticated"):
            page_admin()
        else:
            st.session_state.page = "admin_login"
            st.rerun()
    elif page == "notion":
        page_notion_edit()


if __name__ == "__main__":
    main()
