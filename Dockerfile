# 1. PyTorch公式のPython 3イメージを使用
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 2. 作業ディレクトリの設定
WORKDIR /app

# 3. 依存パッケージのインストール
# バージョンをあえて固定せず、最新の安定版を入れます
RUN pip install --no-cache-dir \
    gym==0.26.2 \
    numpy \
    imageio \
    scipy \
    matplotlib

# 4. ローカルのファイルをコンテナにコピー
COPY . /app

# 5. 実行コマンド（必要に応じて書き換えてください）
CMD ["python", "train.py"]