FROM ubuntu:16.04

# pythonの構築
ENV PYTHON_VERSION 3.8.12
ENV HOME /root
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT $HOME/.pyenv
RUN apt-get update && apt-get upgrade -y \
  && apt-get install -y \
    build-essential \
    curl \
    gcc \
    g++ \
    git \
    libssl-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    llvm \
    make \
    tar \
    tk-dev \
    unzip \
    vim \
    wget \
    xz-utils \
    zlib1g-dev \
  && git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT \
  && $PYENV_ROOT/plugins/python-build/install.sh \
  && /usr/local/bin/python-build -v $PYTHON_VERSION $PYTHON_ROOT \
  && rm -rf $PYENV_ROOT

RUN mkdir -p /root
COPY requirements.txt /root
WORKDIR /root

# 追加ライブラリインストール
RUN pip install -U pip \
  && pip install --no-cache-dir -r requirements.txt

# キャッシュは消す
RUN apt-get autoremove -y \
  && apt-get clean \
  && rm -rf /usr/local/src/*