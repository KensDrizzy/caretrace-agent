# MindBridge 大模型 LoRA 微调全流程

本文档记录 MindBridge 项目中 Qwen2.5-7B 的 LoRA 微调、adapter 导出、模型合并、GGUF 转换、量化以及本地 Ollama 接入流程。按照本文操作，可以从项目数据集训练出 LoRA adapter，并最终得到本项目可直接使用的本地模型文件。

## 1. 最终目标

本流程最终会得到以下文件：



```
mindbridge-qwen2.5-7b-ft-q4\\\_k\\\_m.gguf
```

放到本地项目目录：



```
models/mindbridge-qwen2.5-7b-ft/mindbridge-qwen2.5-7b-ft-q4\\\_k\\\_m.gguf
```

然后用 Ollama 注册为：



```
mindbridge-qwen2.5-7b-ft:latest
```

项目启动后，后端默认调用该模型。

## 2. 数据集说明

本项目用于 LoRA 微调的数据集为：



```
data/lora/psychqa\\\_synthetic.jsonl
```

数据格式为 JSONL，每行一条样本：



```
{"instruction":"分析用户文本情绪，只能输出：正常、焦虑、低落、高风险","input":"最近考试快到了，我总担心自己复习不完，晚上也睡不踏实。","output":"焦虑"}
```

在本地可以先确认：



```
cd MindBridge

wc -l data/lora/psychqa\\\_synthetic.jsonl

head -n 1 data/lora/psychqa\\\_synthetic.jsonl
```

训练时按 9:1 划分训练集和验证集：



```
训练集：约 2160 条

验证集：约 240 条
```

后续训练命令使用 `--split_dataset_ratio 0.1` 自动划分，不需要手动拆成两个文件。

## 3. 云 GPU 环境准备

推荐云 GPU：



```
GPU：A10G 24GB / RTX 4090 24GB / A5000 24GB / A100

系统盘：建议 80GB 以上，最低不要低于 60GB
```

实测 A10G 24GB 可以完成 LoRA 微调。

进入云服务器后，先检查 GPU：



```
nvidia-smi
```

能看到类似：



```
NVIDIA A10G

Memory-Usage: 3MiB / 24564MiB
```

说明 GPU 可用。

## 4. 上传数据集

如果 SSH 可以使用，可以从本地上传整个项目：



```
scp -r MindBridge root@服务器IP:/root/MindBridge
```

如果 SSH 不方便，也可以使用 JupyterLab：



1. 打开云平台里的 JupyterLab

2. 左侧文件区上传 `psychqa_synthetic.jsonl`

3. 打开 JupyterLab 的 Terminal

4. 将文件整理到固定目录

常用整理命令：



```
mkdir -p /root/MindBridge/data/lora

cp /MindBridge/psychqa\\\_synthetic.jsonl /root/MindBridge/data/lora/
```

如果不知道文件被上传到哪里，使用：



```
find /root /MindBridge -name "psychqa\\\_synthetic.jsonl" 2>/dev/null
```

确认数据集：



```
cd /root/MindBridge

wc -l data/lora/psychqa\\\_synthetic.jsonl

head -n 1 data/lora/psychqa\\\_synthetic.jsonl
```

## 5. 安装微调环境

在云服务器 Terminal 中执行：



```
python3 -m venv /root/mindbridge-lora-env

source /root/mindbridge-lora-env/bin/activate

pip install -U pip

pip install -U ms-swift transformers accelerate peft datasets safetensors
```

安装完成后，命令行前面会出现：



```
(mindbridge-lora-env)
```

如果模型下载较慢，可以设置镜像：



```
export HF\\\_ENDPOINT=https://hf-mirror.com
```

实际训练时，`ms-swift` 也可能从 ModelScope 下载模型。

## 6. 开始 LoRA 微调

进入项目目录：



```
cd /root/MindBridge

source /root/mindbridge-lora-env/bin/activate
```

执行训练命令：



```
CUDA\\\_VISIBLE\\\_DEVICES=0 swift sft \\\\

\&#x20; \\--model Qwen/Qwen2.5-7B-Instruct \\\\

\&#x20; \\--dataset /root/MindBridge/data/lora/psychqa\\\_synthetic.jsonl \\\\

\&#x20; \\--tuner\\\_type lora \\\\

\&#x20; \\--lora\\\_rank 8 \\\\

\&#x20; \\--lora\\\_alpha 32 \\\\

\&#x20; \\--target\\\_modules all-linear \\\\

\&#x20; \\--torch\\\_dtype bfloat16 \\\\

\&#x20; \\--num\\\_train\\\_epochs 3 \\\\

\&#x20; \\--per\\\_device\\\_train\\\_batch\\\_size 1 \\\\

\&#x20; \\--gradient\\\_accumulation\\\_steps 16 \\\\

\&#x20; \\--learning\\\_rate 1e-4 \\\\

\&#x20; \\--max\\\_length 512 \\\\

\&#x20; \\--split\\\_dataset\\\_ratio 0.1 \\\\

\&#x20; \\--data\\\_seed 42 \\\\

\&#x20; \\--eval\\\_strategy steps \\\\

\&#x20; \\--eval\\\_steps 100 \\\\

\&#x20; \\--save\\\_steps 100 \\\\

\&#x20; \\--save\\\_total\\\_limit 2 \\\\

\&#x20; \\--logging\\\_steps 5 \\\\

\&#x20; \\--output\\\_dir /root/MindBridge/output/qwen25-7b-mindbridge-lora \\\\

\&#x20; \\--system "你是 MindBridge 校园心理关怀智能体，需要识别学生情绪状态，保持温和、稳定、非评判表达，遇到高风险内容优先保护学生安全。"
```

如果 `bfloat16` 不支持，改成：



```
\\--torch\\\_dtype float16
```

如果显存不够，可以降低上下文长度：



```
\\--max\\\_length 256
```

## 7. 训练日志怎么看

训练前会下载 Qwen2.5-7B-Instruct 原始权重。它通常会分成 4 个 `.safetensors` 文件：



```
model-00001-of-00004.safetensors

model-00002-of-00004.safetensors

model-00003-of-00004.safetensors

model-00004-of-00004.safetensors
```

总大小十几 GB 是正常的，因为这是 FP16/BF16 原始权重。Qwen2.5-7B 约 7.6B 参数，每个参数按 2 bytes 计算，约 15GB。

训练开始后，会看到类似：



```
PeftModelForCausalLM: 7635.8016M Params (20.1851M Trainable)
```

含义：



```
总参数量：约 76 亿

LoRA 可训练参数：约 2018 万

可训练比例：约 0.26%
```

训练进度类似：



```
global\\\_step/max\\\_steps: 102/405

Train: 25%
```

405 步的来源：



```
数据量 2400 条

按 9:1 划分后，训练集约 2160 条，验证集约 240 条

batch\\\_size = 1

gradient\\\_accumulation\\\_steps = 16

每轮约 2160 / 16 = 135 步

num\\\_train\\\_epochs = 3

总步数约 135 \\\* 3 = 405 步
```

训练完成时会看到：



```
Train: 100% 405/405

last\\\_model\\\_checkpoint: /root/MindBridge/output/qwen25-7b-mindbridge-lora/.../checkpoint-405
```

## 8. 找到 LoRA Adapter

训练完成后执行：



```
find /root/MindBridge/output/qwen25-7b-mindbridge-lora -name "adapter\\\_model.safetensors"
```

一般会看到：



```
/root/MindBridge/output/qwen25-7b-mindbridge-lora/v1-20260530-163249/checkpoint-405/adapter\\\_model.safetensors
```

最终使用整个 `checkpoint-405` 文件夹，而不是只使用单个 `.safetensors` 文件。该目录通常包含：



```
adapter\\\_config.json

adapter\\\_model.safetensors

additional\\\_config.json

args.json

README.md

optimizer.pt

rng\\\_state.pth

scheduler.pt

trainer\\\_state.json

training\\\_args.bin
```

真正用于推理 / 合并的核心文件是：



```
adapter\\\_config.json

adapter\\\_model.safetensors
```

## 9. 合并 LoRA 到 Qwen2.5-7B

在云 GPU 上合并，因为基础模型权重已经在云端缓存里。

执行：



```
cd /root/MindBridge

source /root/mindbridge-lora-env/bin/activate

swift export \\\\

\&#x20; \\--adapters /root/MindBridge/output/qwen25-7b-mindbridge-lora/v1-20260530-163249/checkpoint-405 \\\\

\&#x20; \\--merge\\\_lora true \\\\

\&#x20; \\--output\\\_dir /root/MindBridge/output/mindbridge-qwen2.5-7b-ft-hf
```

注意：把路径里的 `v1-20260530-163249/checkpoint-405` 换成你实际生成的最终 checkpoint 路径。

成功时会看到：



```
Successfully merged LoRA and saved in \\\`/root/MindBridge/output/mindbridge-qwen2.5-7b-ft-hf\\\`.
```

合并后的目录是 Hugging Face 格式完整模型，通常包含：



```
config.json

generation\\\_config.json

model-00001-of-00004.safetensors

model-00002-of-00004.safetensors

model-00003-of-00004.safetensors

model-00004-of-00004.safetensors

model.safetensors.index.json

tokenizer.json

tokenizer\\\_config.json

vocab.json

merges.txt
```

## 10. 转换为 GGUF

Ollama 更适合使用 GGUF 文件，因此需要用 `llama.cpp` 转换。

### 10.1 下载 llama.cpp

如果 GitHub 网络正常：



```
cd /root

git clone https://github.com/ggml-org/llama.cpp

cd llama.cpp

pip install -r requirements.txt
```

如果 `git clone` 很慢，可以下载 zip：



```
cd /root

wget -O llama.cpp.zip https://gh-proxy.com/https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.zip
```

如果服务器没有 `unzip`，用 Python 解压：



```
python3 - <<'PY'

import zipfile

from pathlib import Path

zip\\\_path = Path("llama.cpp.zip")

out\\\_dir = Path("/root")

with zipfile.ZipFile(zip\\\_path, "r") as z:

\&#x20;   z.extractall(out\\\_dir)

print("done")

PY
```

整理目录并安装依赖：



```
mv /root/llama.cpp-master /root/llama.cpp

cd /root/llama.cpp

pip install -r requirements.txt
```

如果 pip 下载慢，可以换源：



```
pip install -r requirements.txt \\\\

\&#x20; -i https://mirrors.aliyun.com/pypi/simple \\\\

\&#x20; \\--trusted-host mirrors.aliyun.com
```

### 10.2 转成 F16 GGUF



```
cd /root/llama.cpp

python3 convert\\\_hf\\\_to\\\_gguf.py \\\\

\&#x20; /root/MindBridge/output/mindbridge-qwen2.5-7b-ft-hf \\\\

\&#x20; \\--outfile /root/MindBridge/output/mindbridge-qwen2.5-7b-ft-f16.gguf \\\\

\&#x20; \\--outtype f16
```

转换时会看到：



```
Writing the following files:

/root/MindBridge/output/mindbridge-qwen2.5-7b-ft-f16.gguf

n\\\_tensors = 339, total\\\_size = 15.2G
```

进度条有时会长时间不刷新，可以另开一个 Terminal 看文件大小是否增长：



```
watch -n 10 'date; ls -lh /root/MindBridge/output/mindbridge-qwen2.5-7b-ft-f16.gguf; ps aux | grep convert\\\_hf\\\_to\\\_gguf | grep -v grep'
```

完成后会看到：



```
Model successfully exported to /root/MindBridge/output/mindbridge-qwen2.5-7b-ft-f16.gguf
```

## 11. 量化为 Q4\_K\_M

F16 GGUF 大约 15GB，不适合普通本地机器使用。需要量化成 4-bit。

如果缺少 `cmake`：



```
apt update

apt install -y cmake build-essential
```

如果不能用 apt，也可以：



```
pip install cmake
```

编译量化工具：



```
cd /root/llama.cpp

cmake -B build

cmake --build build --config Release -j
```

执行量化：



```
./build/bin/llama-quantize \\\\

\&#x20; /root/MindBridge/output/mindbridge-qwen2.5-7b-ft-f16.gguf \\\\

\&#x20; /root/MindBridge/output/mindbridge-qwen2.5-7b-ft-q4\\\_k\\\_m.gguf \\\\

\&#x20; Q4\\\_K\\\_M
```

成功时会看到类似：



```
model size = 14526.27 MiB

quant size = 4460.45 MiB
```

确认文件：



```
ls -lh /root/MindBridge/output/mindbridge-qwen2.5-7b-ft-q4\\\_k\\\_m.gguf
```

正常大小约：



```
4.4G - 4.7G
```

量化成功后，可以删除 F16 文件节省云服务器空间：



```
rm -f /root/MindBridge/output/mindbridge-qwen2.5-7b-ft-f16.gguf
```

## 12. 下载最终模型到本地

最终需要下载：



```
/root/MindBridge/output/mindbridge-qwen2.5-7b-ft-q4\\\_k\\\_m.gguf
```

如果使用 JupyterLab，可以在左侧文件浏览器找到该文件，右键下载。

如果 SSH 可用：



```
scp root@服务器IP:/root/MindBridge/output/mindbridge-qwen2.5-7b-ft-q4\\\_k\\\_m.gguf \\\\

\&#x20; ./models/mindbridge-qwen2.5-7b-ft/
```

下载到本地项目后，确认：



```
ls -lh models/mindbridge-qwen2.5-7b-ft/mindbridge-qwen2.5-7b-ft-q4\\\_k\\\_m.gguf
```

## 13. 本地 Ollama 接入

确保本地 `models/mindbridge-qwen2.5-7b-ft/Modelfile` 内容类似：



```
FROM ./mindbridge-qwen2.5-7b-ft-q4\\\_k\\\_m.gguf

SYSTEM """

你是 MindBridge 校园心理关怀智能体，由 Qwen2.5-7B 面向校园心理陪伴场景适配而来。

你需要保持温和、稳定、非评判的表达风格，优先保护学生安全。

当学生表达普通问题时，正常回答，不强行心理测评。

当学生表达情绪困扰时，先共情，再给出具体、可执行的小步骤。

当学生表达自伤、自杀、伤人等高风险信号时，不提供危险细节，鼓励其立即联系身边可信任的人、学校辅导员/心理中心或当地紧急救助。

不要向学生输出后台风险等级、心理报告、评分或诊断结论。

"""

PARAMETER temperature 0.65
```

创建本地 Ollama 模型：



```
cd MindBridge

./scripts/create-finetuned-model.sh
```

检查：



```
/Applications/Ollama.app/Contents/Resources/ollama list
```

应看到：



```
mindbridge-qwen2.5-7b-ft:latest
```

启动项目：



```
./scripts/run-dev.sh
```

浏览器打开：



```
http://localhost:8080
```

## 14. 项目中如何确认调用的是微调模型

默认模型配置：



```
mindbridge:

\&#x20; ai:

\&#x20;   provider: \\\${AI\\\_PROVIDER:ollama}

\&#x20;   ollama:

\&#x20;     model: \\\${OLLAMA\\\_MODEL:mindbridge-qwen2.5-7b-ft:latest}
```

启动脚本默认模型：



```
OLLAMA\\\_MODEL="\\\${OLLAMA\\\_MODEL:-mindbridge-qwen2.5-7b-ft:latest}"
```

Ollama 客户端请求体会把该模型名传给 `/api/chat`：



```
"model", model
```

整体链路：



```
application.yml / run-dev.sh

\&#x20;       ↓

mindbridge-qwen2.5-7b-ft:latest

\&#x20;       ↓

OllamaAiClient

\&#x20;       ↓

Ollama

\&#x20;       ↓

Modelfile

\&#x20;       ↓

mindbridge-qwen2.5-7b-ft-q4\\\_k\\\_m.gguf
```

## 15. 常见问题

### 15.1 `mv psychqa_synthetic.jsonl ...` 提示没有文件

说明当前目录没有该文件。先查找：



```
find /root /MindBridge -name "psychqa\\\_synthetic.jsonl" 2>/dev/null
```

找到后再复制到：



```
/root/MindBridge/data/lora/
```

### 15.2 `ValueError: remaining_argv: ['--train_type', 'lora']`

新版 `ms-swift` 使用：



```
\\--tuner\\\_type lora
```

不要使用：



```
\\--train\\\_type lora
```

### 15.3 为什么下载 4 个权重文件，总共十几 GB

这是正常现象。LoRA 微调虽然只训练 adapter，但基础模型仍然要参与前向计算，所以必须加载完整 Qwen2.5-7B 权重。FP16/BF16 权重大约 15GB。

### 15.4 为什么训练命令要划分验证集

本项目文档统一按 9:1 划分训练集和验证集，这样训练时可以看到验证集上的评估情况，也更适合写实验报告或项目说明。

如果不加 `--split_dataset_ratio 0.1`，2400 条会全部进入训练集，总步数会变成约 450 步；本文档按 9:1 划分后，总步数约为 405 步。

### 15.5 GGUF 转换一直停在 0% 或 7%

先检查文件大小是否增长：



```
ls -lh /root/MindBridge/output/mindbridge-qwen2.5-7b-ft-f16.gguf

ps aux | grep convert\\\_hf\\\_to\\\_gguf | grep -v grep
```

如果文件在增长，说明还在写。云服务器系统盘 IO 慢时，进度条可能长时间不刷新。

### 15.6 `llama_model_quantize: failed ... iostream error`

大概率是磁盘空间不足。检查：



```
df -h
```

清理残缺文件和不再需要的目录：



```
rm -f /root/MindBridge/output/mindbridge-qwen2.5-7b-ft-q4\\\_k\\\_m.gguf

rm -rf /root/MindBridge/output/mindbridge-qwen2.5-7b-ft-hf

rm -rf /root/.cache/modelscope/hub/models/Qwen/Qwen2\\\_\\\_\\\_5-7B-Instruct

rm -rf /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct

df -h
```

然后重新量化。

### 15.7 `bash: cmake: 未找到命令`

安装：



```
apt update

apt install -y cmake build-essential
```

或：



```
pip install cmake
```

### 15.8 `bash: unzip: 未找到命令`

用 Python 解压 zip：



```
python3 - <<'PY'

import zipfile

from pathlib import Path

with zipfile.ZipFile(Path("llama.cpp.zip"), "r") as z:

\&#x20;   z.extractall(Path("/root"))

print("done")

PY
```

### 15.9 pip 下载很慢

换源：



```
pip install -r requirements.txt \\\\

\&#x20; -i https://mirrors.aliyun.com/pypi/simple \\\\

\&#x20; \\--trusted-host mirrors.aliyun.com
```

或单独安装慢的包：



```
pip install numpy==1.26.4 \\\\

\&#x20; -i https://mirrors.aliyun.com/pypi/simple \\\\

\&#x20; \\--trusted-host mirrors.aliyun.com
```

## 16. 云服务器释放

确认以下文件已经下载到本地：



```
models/mindbridge-qwen2.5-7b-ft/mindbridge-qwen2.5-7b-ft-q4\\\_k\\\_m.gguf
```

并确认本地 Ollama 已经创建：



```
/Applications/Ollama.app/Contents/Resources/ollama list
```

看到：



```
mindbridge-qwen2.5-7b-ft:latest
```

即可在云平台释放实例。注意：如果云平台有 “关机” 和 “释放”，通常应选择 “释放”，避免继续计费。

## 17. 产物清单

云端训练产物：



```
/root/MindBridge/output/qwen25-7b-mindbridge-lora/.../checkpoint-405

/root/MindBridge/output/mindbridge-qwen2.5-7b-ft-hf

/root/MindBridge/output/mindbridge-qwen2.5-7b-ft-f16.gguf

/root/MindBridge/output/mindbridge-qwen2.5-7b-ft-q4\\\_k\\\_m.gguf
```

最终交付 / 运行只需要：



```
models/mindbridge-qwen2.5-7b-ft/mindbridge-qwen2.5-7b-ft-q4\\\_k\\\_m.gguf

models/mindbridge-qwen2.5-7b-ft/Modelfile

scripts/create-finetuned-model.sh

scripts/run-dev.sh
```

如果需要证明训练过程，可以额外保留：



```
checkpoint-405/

logging.jsonl

args.json
```