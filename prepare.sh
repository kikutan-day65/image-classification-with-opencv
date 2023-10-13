#!/bin/bash

# ループを回すディレクトリパス
input_directory="./pdfs_image_classification_task/pdfs/"

# ループして.pdfファイルを処理
for i in {1..22}
do
  # 出力ディレクトリを作成
  output_directory="test$i"
  mkdir -p "$output_directory"

  # pdfimagesコマンドを実行して画像を抽出
  pdfimages -j "$input_directory/$i.pdf" "$output_directory/test"

  # 結果フォルダを作成
  result_directory="result$i"
  mkdir -p "$result_directory"

  # 結果フォルダ内にdiagram、image、textフォルダを作成
  mkdir -p "$result_directory/diagram"
  mkdir -p "$result_directory/image"
  mkdir -p "$result_directory/text"

done
