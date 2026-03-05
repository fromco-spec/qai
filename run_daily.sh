#!/bin/bash
# run_daily.sh
# launchd から毎朝7時に呼ばれる自動更新スクリプト
# （日本語パスを含むため、シンボリックリンク ~/qai_app 経由で実行）

PYTHON="/Library/Developer/CommandLineTools/usr/bin/python3"
WORKDIR="/Users/oritonoboru/qai_app"

echo "======================================"
echo " Q&Ai 自動更新開始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================"

cd "$WORKDIR" || { echo "ERROR: ディレクトリ移動失敗 $WORKDIR"; exit 1; }

echo "[Step1] サマリー自動生成..."
$PYTHON "$WORKDIR/auto_summarize.py"
STATUS1=$?
echo "Step1 終了コード: $STATUS1"

echo "[Step2] 知識ベース更新..."
$PYTHON "$WORKDIR/fetch_knowledge.py"
STATUS2=$?
echo "Step2 終了コード: $STATUS2"

echo "======================================"
echo " 完了: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================"

exit $((STATUS1 + STATUS2))
