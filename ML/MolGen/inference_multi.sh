#!/bin/bash

# 10回のループを設定
for i in {1..10}
do
    echo "実行回数: $i"
    bash inference.sh
done

echo "スクリプトの実行が完了しました。"