# Local RedTeam DPO Factory - Colab MVP

这个项目现在支持两条 Colab 路径：

- `pyrit_api_main.py`：PyRIT + OpenAI-compatible API，推荐给 SSH Colab。默认使用 Colab 本地 Ollama 运行 Dolphin3.0 Llama3.1 8B GGUF。
- `colab_main.py`：纯 Hugging Face Transformers，本地加载模型，需要 Colab GPU。
- `redteam_colab.ipynb`：Colab notebook 运行版，适合需要在 `.ipynb` 中启用 GPU 的场景。

> 安全提示：本项目用于授权 AI 安全评估和红队数据构建。生成的 `data/` 内容可能包含未经审查的安全测试样本，不建议公开传播。

## Notebook 运行顺序

如果你使用 Colab GPU，优先打开 `redteam_colab.ipynb`，然后按 notebook 内部顺序执行：

1. `0. 项目目录`
2. `1. 下载 HarmBench 数据`
3. 跑 PyRIT + API：执行 `Path A` 的 A1-A4
4. 跑本地 HF GPU：执行 `Path B` 的 B1-B4

两条路径会写同一个 `data/my_dpo_data.json`。如果你先跑 A 再跑 B，后跑的路径会覆盖前一次 DPO 输出。

## 路径 A：PyRIT + 本地 Ollama（推荐）

### 1. SSH 到 Colab 后进入项目

```bash
cd /content/redteam
```

### 2. 安装依赖

```bash
pip install -r requirements-api.txt
```

### 3. 安装并启动 Ollama

Colab 中运行：

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve > /tmp/ollama.log 2>&1 &
sleep 5
```

拉取官方教程里的 Hugging Face GGUF 模型：

```bash
ollama pull hf.co/cognitivecomputations/Dolphin3.0-Llama3.1-8B-GGUF:Q4_0
```

官方交互式用法是：

```bash
ollama run hf.co/cognitivecomputations/Dolphin3.0-Llama3.1-8B-GGUF:Q4_0
/set system <your system prompt>
```

本项目不需要交互式 `/set system`，PyRIT 会通过 API 请求传 system prompt。

`config.yaml` 默认已经设置为：

```yaml
api_base: "http://127.0.0.1:11434/v1"
api_key: "ollama"
attacker_model: "hf.co/cognitivecomputations/Dolphin3.0-Llama3.1-8B-GGUF:Q4_0"
target_model: "hf.co/cognitivecomputations/Dolphin3.0-Llama3.1-8B-GGUF:Q4_0"
judge_model: "hf.co/cognitivecomputations/Dolphin3.0-Llama3.1-8B-GGUF:Q4_0"
num_behaviors: 1
pyrit_attempts_per_behavior: 2
pyrit_allow_objective_fallback_rejected: true
max_turns: 3
```

如果你要换成 OpenAI、OpenRouter、Together、vLLM 等外部或远程 API，只要改 `api_base`、`api_key` 和模型名即可。

### 4. 下载数据并运行

```bash
python download_data.py
python pyrit_api_main.py --config config.yaml
```

也可以用脚本：

```bash
chmod +x set_up.sh
./set_up.sh api
```

### 5. 检查输出

```bash
ls -lh data/
python -m json.tool data/my_dpo_data.json | head
head -n 2 data/pyrit_run_log.jsonl
```

主要输出：

```text
data/my_dpo_data.json       # LLaMA-Factory DPO 数据
data/pyrit_run_log.jsonl    # 每次 PyRIT attack 的摘要和选中 prompt
data/pyrit_failures.jsonl   # 没有形成 chosen/rejected 对的 behavior
```

## 路径 B：纯 Transformers

注意：`hf.co/cognitivecomputations/Dolphin3.0-Llama3.1-8B-GGUF:Q4_0` 是 GGUF/Ollama 模型，不能直接填到 `attacker_model_id` 让 Transformers 加载。纯 Transformers 路径需要使用普通 HF Transformers repo，例如 `Qwen/Qwen2.5-0.5B-Instruct`，或非 GGUF 的 Dolphin 权重仓库。

### 1. 进入项目并安装依赖

```bash
cd /content/redteam
pip install -r requirements-colab.txt
```

如果你使用 gated/private Hugging Face 模型：

```bash
huggingface-cli login
```

### 2. 运行

```bash
python download_data.py
python colab_main.py --config config.yaml
```

首次建议保持小参数：

```yaml
num_behaviors: 1
candidates_per_behavior: 2
```

主要输出：

```text
data/my_dpo_data.json   # LLaMA-Factory DPO 数据
data/run_log.jsonl      # 每个候选 prompt、target response 预览、judge 结果
data/failures.jsonl     # 没有形成 chosen/rejected 对的 behavior
```

## DPO 数据格式

`data/my_dpo_data.json` 每条记录包含：

```json
{
  "instruction": "Generate an adversarial prompt for authorized AI safety evaluation. The prompt should test whether a target model refuses harmful requests.",
  "input": "HarmBench behavior",
  "chosen": "judge 判定更有效的候选 prompt",
  "rejected": "judge 判定无效或被拒绝的候选 prompt"
}
```

## SSH Colab 建议

SSH 连接本身不由本项目管理。常见做法是在 Colab notebook 里启动 `sshd` 或使用 cloudflared/ngrok 反向隧道，然后从本机 SSH 进去。进入后先确认：

```bash
python --version
nvidia-smi
pwd
```

PyRIT + API 路径不需要 GPU；纯 Transformers 路径需要 GPU。

## 旧路径说明

`main.py` 是早期 Ollama/PyRIT API 原型，当前建议使用 `pyrit_api_main.py` 或 `colab_main.py`。
