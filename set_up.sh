#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-api}"

echo "========================================="
echo " Local RedTeam DPO Factory - Colab MVP"
echo " mode: ${MODE}"
echo "========================================="

case "${MODE}" in
  api)
    echo "[1/3] Installing PyRIT/API dependencies..."
    python -m pip install -r requirements-api.txt

    if command -v ollama >/dev/null 2>&1; then
      echo "[ollama] Starting local Ollama server if needed..."
      if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        nohup ollama serve >/tmp/ollama.log 2>&1 &
        sleep 5
      fi
      echo "[ollama] Pulling Dolphin3.0 Llama3.1 8B GGUF Q4_0..."
      ollama pull hf.co/cognitivecomputations/Dolphin3.0-Llama3.1-8B-GGUF:Q4_0
    else
      echo "[ollama] Not installed. In Colab run first:"
      echo "         curl -fsSL https://ollama.com/install.sh | sh"
      echo "         ollama serve > /tmp/ollama.log 2>&1 &"
      echo "         ollama pull hf.co/cognitivecomputations/Dolphin3.0-Llama3.1-8B-GGUF:Q4_0"
    fi

    echo "[2/3] Downloading HarmBench behaviors..."
    python download_data.py

    echo "[3/3] Running PyRIT + API DPO generation..."
    python pyrit_api_main.py --config config.yaml
    ;;

  transformers)
    echo "[1/3] Installing Transformers dependencies..."
    python -m pip install -r requirements-colab.txt

    echo "[2/3] Downloading HarmBench behaviors..."
    python download_data.py

    echo "[3/3] Running local Transformers DPO generation..."
    python colab_main.py --config config.yaml
    ;;

  *)
    echo "Unknown mode: ${MODE}"
    echo "Usage: ./set_up.sh [api|transformers]"
    exit 2
    ;;
esac

echo "========================================="
echo "Done. Check data/my_dpo_data.json"
echo "========================================="
