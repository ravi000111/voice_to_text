#!/usr/bin/env bash
set -e
wget -qO localai.tgz \
  https://github.com/go-skynet/LocalAI/releases/latest/download/localai-linux-amd64.tar.gz
tar -xzf localai.tgz && mv localai /usr/local/bin/

localai pull mistral-7b-instruct-v0.2-q4
localai serve --models-path ~/.localai/models --host 0.0.0.0 --port 8080 &

streamlit run app.py --server.port 8501 --server.address 0.0.0.0
