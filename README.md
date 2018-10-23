# tenma

開発環境には[Docker](https://www.docker.com/)を利用します

```
# イメージ作成(Dockerfileがあるディレクトリで実行)
$ docker build ./ -t tenma

# コンテナ作成
$ docker run -v $PWD:/work -itd --name tenma_01 tenma

# Docker内でpython実行
$ docker exec -ti tenma_01 python <hogehoge.py>

# テストを実行
$ docker exec -it tenma_01 python -m unittest discover tenma/tests

# コンテナをsshのように対話的に利用する
$ docker exec -ti tenma_01 bash

# コンテナの停止
$ docker stop tenma_01

# コンテナの開始
$ docker start tenma_01
```

モジュールを追加する際には、requirements.txtに記載してください
```
# ファイルを編集
$ vim requirements.txt

# コンテナを再ビルド（Dockerfileでpipを実行しています）
# 必要に応じて、イメージとコンテナを削除
$ docker rm -f tenma_01
$ docker build ./ -t tenma
$ docker run -v $PWD:/work -itd --name tenma_01 tenma
$ docker exec -ti tenma_01 pip list
```

ユニットテスト
```
# 個々のテストを実行
$ docker exec -it tenma_01 python -m unittest tenma/tests/test_hello.py

# ディレクトリ配下のテストを実行
# test_<hoge>.py が対象
$ docker exec -it tenma_01 python -m unittest discover tenma/tests

# 全てのテストを実行
$ docker exec -it tenma_01 bash /work/run_test.sh
```
