# MindBridge 大模型 LoRA 微调全流程（Qwen3-14B + SoulChatCorpus）

本文档记录 MindBridge 项目中 Qwen3-14B-Instruct 的 LoRA 微调、adapter 导出、模型合并、GGUF 转换、量化以及本地 Ollama 接入流程。按照本文操作，可以用 SoulChatCorpus 25 万条多轮对话数据训练出 LoRA adapter，并最终得到本项目可直接使用的本地模型文件。

## 1. 最终目标

本流程最终会得到以下文件：



```
mindbridge-qwen3-14b-ft-q4\_k\_m.gguf
```

放到本地项目目录：



```
models/mindbridge-qwen3-14b-ft/mindbridge-qwen3-14b-ft-q4\_k\_m.gguf
```

然后用 Ollama 注册为：



```
mindbridge-qwen3-14b-ft:latest
```

项目启动后，后端默认调用该模型。

### 关键参数一览



| 项目             | 值                                                |
| -------------- | ------------------------------------------------ |
| 基座模型           | Qwen/Qwen3-14B-Instruct                          |
| 微调方式           | LoRA（rank=16, alpha=32, all-linear）              |
| 训练数据           | SoulChatCorpus train.json（245,242 条多轮对话）         |
| 数据格式           | sharegpt（messages: user/assistant 交替）            |
| GPU            | 4090D-48G-DDR5（48G 显存，22 核 CPU，120G 内存，100G 系统盘） |
| max\_length    | 2048                                             |
| batch\_size    | 2（gradient\_accumulation\_steps=8，有效 batch=16）   |
| epochs         | 2                                                |
| learning\_rate | 5e-5                                             |
| 预估训练时间         | 约 15-25 小时                                       |
| 预估费用           | 约 ¥37-62（¥2.48 / 时，按时计费）                         |

### 磁盘空间说明

4090D-48G-DDR5 系统盘为 100GB，空间相对充裕。14B 模型训练各阶段空间需求：



| 阶段                | 占用     | 累计                 |
| ----------------- | ------ | ------------------ |
| 基座模型缓存（BF16）      | \~28GB | 28GB               |
| 训练数据 + adapter 输出 | \~3GB  | 31GB               |
| 合并后 HF 格式模型       | \~28GB | 59GB               |
| F16 GGUF          | \~28GB | 87GB（峰值，仍在 100G 内） |
| Q4\_K\_M GGUF     | \~9GB  | 最终仅保留 9GB          |

100GB 可以容纳所有阶段的峰值，但仍建议在各阶段结束后清理中间文件，避免意外。

## 2. 数据集说明

本项目用于 LoRA 微调的数据集为 SoulChatCorpus，已经过清洗、去重、按话题分层划分，存放在：



```
data/dataset/
```

各文件说明：



| 文件                                 | 条数      | 大小    | 用途                      |
| ---------------------------------- | ------- | ----- | ----------------------- |
| train.json                         | 245,242 | 733MB | 训练集（95%），训练时上传到服务器      |
| val.json                           | 6,454   | 19MB  | 验证集（2.5%），本地备用，不需要上传    |
| test.json                          | 6,454   | 19MB  | 测试集（2.5%），训练完成后评估用，建议上传 |
| campus\_eval.json                  | 3,180   | 10MB  | 校园场景专项评估集，建议上传          |
| SoulChatCorpus-sft-multi-Turn.json | 258,353 | 907MB | 原始数据，不需要上传              |

每条数据为多轮对话，格式如下：



```
{

&#x20; "id": 93401,

&#x20; "topic": "成长",

&#x20; "messages": \[

&#x20;   {"role": "user", "content": "不知道是因为什么，我经历了长达一年多的自我否定。"},

&#x20;   {"role": "assistant", "content": "你好，听到你说这些，我能理解你经历了这么长时间的自我否定..."},

&#x20;   {"role": "user", "content": "是的，我一直以来都对自己充满了负面情绪..."},

&#x20;   {"role": "assistant", "content": "很多人都会在压力下负面情绪泛滥..."}

&#x20; ]

}
```

训练时使用 `--split_dataset_ratio 0.1`，ms-swift 会自动从 train.json 中划 10%（约 2.4 万条）做验证集，不需要手动上传 val.json。

## 3. 云 GPU 环境准备

### 3.1 创建实例

打开 AI Gate 平台：



```
https://waas.aigate.cc
```

选择配置：



* 区域：华东二区（或就近选择）

* 算力产品：4090D-48G-DDR5（¥2.48 / 时）

* 租赁数量：1

* 系统镜像：选择带 PyTorch 的镜像（预装 CUDA、PyTorch、Miniconda）

* 预装框架：勾选 pytorch、Miniconda

* 计费方式：按时

右侧确认规格：



```
CPU: 22Core\*1

显存: 48G\*1

内存: 120G\*1

系统盘: 100G\*1
```

点击「创建实例」，等待 1-3 分钟状态变为「运行中」。

### 3.2 进入服务器

方式 A：JupyterLab（无需配置 SSH）



1. 在 AI Gate 控制台找到刚创建的实例

2. 点击「JupyterLab」或「打开」按钮

3. 点击顶部菜单 File → New → Terminal，打开命令行终端

方式 B：SSH（适合大文件传输）

在 AI Gate 控制台找到实例的 SSH 连接信息（主机、端口、密码），然后在本地 Mac 终端执行：



```
ssh -p 端口号 root@服务器IP
```

输入密码后进入服务器。首次连接提示确认主机指纹时输入 yes。

### 3.3 检查 GPU 和磁盘

进入服务器后，先检查 GPU：



```
nvidia-smi
```

能看到类似：



```
NVIDIA 4090D

Memory-Usage: 3MiB / 49140MiB
```

说明 48G GPU 可用。

检查磁盘：



```
df -h /
```

应看到 `/` 分区约 100GB，已用较少（5-10GB）。

检查 Python：



```
python3 --version

which python3
```

## 4. 上传数据集

只需要上传 train.json（必须）、test.json 和 campus\_eval.json（可选，评估用），不需要上传整个项目，也不需要上传原始数据文件。

### 4.1 方式 A：scp 上传（推荐）

在本地 Mac 终端（不是服务器终端）执行：



```
\# 先在服务器上创建目录

ssh -p 端口号 root@服务器IP "mkdir -p /root/MindBridge/data/dataset"

\# 上传训练数据（733MB，约 3-10 分钟，取决于网速）

scp -P 端口号 /Users/lqf/projects/agentcode/mindbridge-py/data/dataset/train.json root@服务器IP:/root/MindBridge/data/dataset/

\# 上传评估数据（可选，文件很小，建议一起传）

scp -P 端口号 /Users/lqf/projects/agentcode/mindbridge-py/data/dataset/test.json root@服务器IP:/root/MindBridge/data/dataset/

scp -P 端口号 /Users/lqf/projects/agentcode/mindbridge-py/data/dataset/campus\_eval.json root@服务器IP:/root/MindBridge/data/dataset/
```

### 4.2 方式 B：JupyterLab 上传

如果没有 SSH，在 JupyterLab 左侧文件浏览器中：



1. 导航到 `/root/MindBridge/data/dataset/`（没有就新建目录）

2. 点击左上角上传按钮（向上箭头图标）

3. 选择本地的 train.json、test.json、campus\_eval.json

4. 等待上传完成（733MB 约 5-15 分钟）

### 4.3 确认数据集

在服务器终端执行：



```
cd /root/MindBridge/data/dataset

ls -lh
```

应看到：



```
train.json        733M

test.json          19M

campus\_eval.json   10M
```

验证 train.json 条数（应该是 245242）：



```
python3 -c "import json; data=json.load(open('train.json')); print('条数:', len(data)); print('第一条topic:', data\[0]\['topic']); print('第一条轮数:', len(data\[0]\['messages']))"
```

应输出：



```
条数: 245242

第一条topic: 婚恋

第一条轮数: 12
```

如果条数不对，说明上传不完整，重新上传。

## 5. 安装微调环境

在云服务器 Terminal 中执行：



```
python3 -m venv /root/mindbridge-14b-env

source /root/mindbridge-14b-env/bin/activate

pip install -U pip
```

安装 ms-swift 及依赖：



```
pip install -U ms-swift transformers accelerate peft datasets safetensors sentencepiece
```

如果下载慢，换阿里云源：



```
pip install -U ms-swift transformers accelerate peft datasets safetensors sentencepiece \\

&#x20; -i https://mirrors.aliyun.com/pypi/simple \\

&#x20; \--trusted-host mirrors.aliyun.com
```

验证安装：



```
swift --version
```

安装完成后，命令行前面会出现：



```
(mindbridge-14b-env)
```

设置模型下载镜像（加速 28GB 基座模型下载）：



```
export MODELSCOPE\_CACHE=/root/.cache/modelscope

export HF\_ENDPOINT=https://hf-mirror.com
```

写入 bashrc 避免每次重新设置：



```
echo 'export MODELSCOPE\_CACHE=/root/.cache/modelscope' >> \~/.bashrc

echo 'export HF\_ENDPOINT=https://hf-mirror.com' >> \~/.bashrc

source \~/.bashrc
```

## 6. 开始 LoRA 微调

### 6.1 进入工作目录



```
cd /root/MindBridge

source /root/mindbridge-14b-env/bin/activate
```

### 6.2 训练命令（前台运行，用于测试）



```
CUDA\_VISIBLE\_DEVICES=0 swift sft \\

&#x20; \--model Qwen/Qwen3-14B-Instruct \\

&#x20; \--dataset /root/MindBridge/data/dataset/train.json \\

&#x20; \--dataset\_type sharegpt \\

&#x20; \--tuner\_type lora \\

&#x20; \--lora\_rank 16 \\

&#x20; \--lora\_alpha 32 \\

&#x20; \--target\_modules all-linear \\

&#x20; \--torch\_dtype bfloat16 \\

&#x20; \--num\_train\_epochs 2 \\

&#x20; \--per\_device\_train\_batch\_size 2 \\

&#x20; \--gradient\_accumulation\_steps 8 \\

&#x20; \--learning\_rate 5e-5 \\

&#x20; \--max\_length 2048 \\

&#x20; \--split\_dataset\_ratio 0.1 \\

&#x20; \--data\_seed 42 \\

&#x20; \--eval\_strategy steps \\

&#x20; \--eval\_steps 500 \\

&#x20; \--save\_steps 500 \\

&#x20; \--save\_total\_limit 2 \\

&#x20; \--logging\_steps 10 \\

&#x20; \--output\_dir /root/MindBridge/output/qwen3-14b-mindbridge-lora \\

&#x20; --system "你是 MindBridge 校园心理关怀智能体，擅长共情倾听、引导倾诉和温和建议，遇到高风险内容优先保护学生安全。"
```

### 6.3 后台运行（正式训练用这个）

训练时间长达十几小时，必须用后台运行，避免 SSH/JupyterLab 断开导致训练中断。



```
cd /root/MindBridge

source /root/mindbridge-14b-env/bin/activate

nohup bash -c '

CUDA\_VISIBLE\_DEVICES=0 swift sft \\

&#x20; \--model Qwen/Qwen3-14B-Instruct \\

&#x20; \--dataset /root/MindBridge/data/dataset/train.json \\

&#x20; \--dataset\_type sharegpt \\

&#x20; \--tuner\_type lora \\

&#x20; \--lora\_rank 16 \\

&#x20; \--lora\_alpha 32 \\

&#x20; \--target\_modules all-linear \\

&#x20; \--torch\_dtype bfloat16 \\

&#x20; \--num\_train\_epochs 2 \\

&#x20; \--per\_device\_train\_batch\_size 2 \\

&#x20; \--gradient\_accumulation\_steps 8 \\

&#x20; \--learning\_rate 5e-5 \\

&#x20; \--max\_length 2048 \\

&#x20; \--split\_dataset\_ratio 0.1 \\

&#x20; \--data\_seed 42 \\

&#x20; \--eval\_strategy steps \\

&#x20; \--eval\_steps 500 \\

&#x20; \--save\_steps 500 \\

&#x20; \--save\_total\_limit 2 \\

&#x20; \--logging\_steps 10 \\

&#x20; \--output\_dir /root/MindBridge/output/qwen3-14b-mindbridge-lora \\

&#x20; \--system "你是 MindBridge 校园心理关怀智能体，擅长共情倾听、引导倾诉和温和建议，遇到高风险内容优先保护学生安全。"

' > /root/MindBridge/train.log 2>&1 &
```

查看进程是否启动：



```
ps aux | grep "swift sft" | grep -v grep
```

### 6.4 参数说明



| 参数                                | 值                       | 说明                                   |
| --------------------------------- | ----------------------- | ------------------------------------ |
| --model                           | Qwen/Qwen3-14B-Instruct | 基座模型，ms-swift 从 ModelScope 自动下载      |
| --dataset                         | train.json 路径           | 训练数据，24.5 万条多轮对话                     |
| --dataset\_type                   | sharegpt                | 数据格式为 messages 数组（user/assistant 交替） |
| --tuner\_type                     | lora                    | LoRA 微调                              |
| --lora\_rank                      | 16                      | LoRA 秩，14B 模型用 16 表达力更强              |
| --lora\_alpha                     | 32                      | LoRA alpha，通常为 rank 的 2 倍            |
| --target\_modules                 | all-linear              | 对所有线性层加 LoRA                         |
| --torch\_dtype                    | bfloat16                | 训练精度，4090D 支持 BF16                   |
| --num\_train\_epochs              | 2                       | 训练 2 轮（25 万条大数据，2 轮足够）               |
| --per\_device\_train\_batch\_size | 2                       | 每卡 batch=2（48G 显存可以）                 |
| --gradient\_accumulation\_steps   | 8                       | 梯度累积 8 步，有效 batch=16                 |
| --learning\_rate                  | 5e-5                    | 学习率，大数据集 + 大模型用稍低更稳                  |
| --max\_length                     | 2048                    | 最大序列长度，多轮对话平均 774 字符，2048 足够         |
| --split\_dataset\_ratio           | 0.1                     | 自动从训练集划 10% 做验证（约 2.4 万条）            |
| --eval\_steps                     | 500                     | 每 500 步评估一次                          |
| --save\_steps                     | 500                     | 每 500 步保存一次 checkpoint               |
| --save\_total\_limit              | 2                       | 最多保留 2 个 checkpoint（节省空间）            |
| --system                          | 校园心理关怀提示词               | 系统提示词，加入每条训练样本                       |

如果 bfloat16 不支持，改成 `--torch_dtype float16`。

如果显存不够（48G 一般不会），可以降低 batch\_size 和上下文长度：



```
\--per\_device\_train\_batch\_size 1 \\

\--gradient\_accumulation\_steps 16 \\

\--max\_length 1024
```

### 6.5 监控训练

查看实时日志：



```
tail -f /root/MindBridge/train.log
```

按 Ctrl+C 退出查看（不会停止训练）。

查看 GPU 使用情况：



```
nvidia-smi
```

应看到显存占用约 35-42GB（14B BF16 权重 28GB + 激活 + LoRA + 优化器状态）。

查看磁盘空间（训练期间定期检查）：



```
df -h /
```

## 7. 训练日志怎么看

### 7.1 模型下载阶段

训练前会下载 Qwen3-14B-Instruct 原始权重，通常分成 8 个 `.safetensors` 文件，总大小约 28GB。从 ModelScope 镜像下载约 5-15 分钟。



```
Downloading model: Qwen/Qwen3-14B-Instruct

model-00001-of-00008.safetensors: 100%|████| 3.85G/3.85G

...
```

### 7.2 模型加载阶段

训练开始后，会看到类似：



```
PeftModelForCausalLM: 14000.0000M Params (80.0000M Trainable)
```

含义：



```
总参数量：约 140 亿

LoRA 可训练参数：约 8000 万

可训练比例：约 0.57%
```

### 7.3 训练进度

训练进度类似：



```
global\_step/max\_steps: 500/27590

Train: 1.8%

loss: 2.3456, learning\_rate: 4.99e-05, epoch: 0.04
```

总步数的来源：



```
数据量 245,242 条

按 9:1 划分后，训练集约 220,718 条，验证集约 24,524 条

batch\_size = 2

gradient\_accumulation\_steps = 8

有效 batch = 2 × 8 = 16

每轮约 220718 / 16 ≈ 13,795 步

num\_train\_epochs = 2

总步数约 13795 × 2 ≈ 27,590 步
```

每步耗时约 1.5-3 秒（4090D 48G），总训练时间约 12-23 小时。

### 7.4 评估与保存

每 500 步会在验证集上评估并保存 checkpoint：



```
eval\_loss: 1.8765

Saving model checkpoint to /root/MindBridge/output/qwen3-14b-mindbridge-lora/v1-xxx/checkpoint-500
```

save\_total\_limit=2 意味着只保留最近 2 个 checkpoint，旧的自动删除。

### 7.5 训练完成

训练完成时会看到：



```
Train: 100% 27590/27590

last\_model\_checkpoint: /root/MindBridge/output/qwen3-14b-mindbridge-lora/v1-xxx/checkpoint-27590
```

## 8. 找到 LoRA Adapter

训练完成后执行：



```
find /root/MindBridge/output/qwen3-14b-mindbridge-lora -name "adapter\_model.safetensors"
```

一般会看到：



```
/root/MindBridge/output/qwen3-14b-mindbridge-lora/v1-20260901-xxxxxx/checkpoint-27590/adapter\_model.safetensors
```

最终使用整个 checkpoint-27590 文件夹，而不是只使用单个 `.safetensors` 文件。真正用于推理 / 合并的核心文件是：



```
adapter\_config.json

adapter\_model.safetensors
```

## 9. 合并 LoRA 到 Qwen3-14B

在云 GPU 上合并，因为基础模型权重已经在云端缓存里。

执行：



```
cd /root/MindBridge

source /root/mindbridge-14b-env/bin/activate

swift export \\

&#x20; \--adapters /root/MindBridge/output/qwen3-14b-mindbridge-lora/v1-20260901-xxxxxx/checkpoint-27590 \\

&#x20; \--merge\_lora true \\

&#x20; \--output\_dir /root/MindBridge/output/mindbridge-qwen3-14b-ft-hf
```

注意：把路径里的 `v1-20260901-xxxxxx/checkpoint-27590` 换成你实际生成的最终 checkpoint 路径。

成功时会看到：



```
Successfully merged LoRA and saved in \`/root/MindBridge/output/mindbridge-qwen3-14b-ft-hf\`.
```

合并后的目录是 Hugging Face 格式完整模型，约 28GB。

### 空间清理（推荐）

合并完成后，基座模型缓存和训练 checkpoint 都不再需要，建议删除以释放空间：



```
\# 删除基座模型缓存（释放 \~28GB）

rm -rf /root/.cache/modelscope/hub/models/Qwen/Qwen3-14B-Instruct

rm -rf /root/.cache/huggingface/hub/models--Qwen--Qwen3-14B-Instruct

\# 删除训练输出（adapter checkpoint，释放 \~3-5GB）

rm -rf /root/MindBridge/output/qwen3-14b-mindbridge-lora

\# 检查空间

df -h /
```

## 10. 转换为 GGUF

Ollama 更适合使用 GGUF 文件，因此需要用 llama.cpp 转换。

### 10.1 下载 llama.cpp

如果 GitHub 网络正常：



```
cd /root

git clone https://github.com/ggml-org/llama.cpp

cd llama.cpp

pip install -r requirements.txt
```

如果 git clone 很慢，下载 zip：



```
cd /root

wget -O llama.cpp.zip https://gh-proxy.com/https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.zip

python3 -c "

import zipfile

with zipfile.ZipFile('llama.cpp.zip', 'r') as z:

&#x20;   z.extractall('/root')

print('done')

"

mv /root/llama.cpp-master /root/llama.cpp

cd /root/llama.cpp

pip install -r requirements.txt
```

如果 pip 下载慢，换源：



```
pip install -r requirements.txt \\

&#x20; -i https://mirrors.aliyun.com/pypi/simple \\

&#x20; \--trusted-host mirrors.aliyun.com
```

### 10.2 转成 F16 GGUF



```
cd /root/llama.cpp

python3 convert\_hf\_to\_gguf.py \\

&#x20; /root/MindBridge/output/mindbridge-qwen3-14b-ft-hf \\

&#x20; \--outfile /root/MindBridge/output/mindbridge-qwen3-14b-ft-f16.gguf \\

&#x20; \--outtype f16
```

转换过程约 10-20 分钟（14B 模型，28GB）。完成后会看到：



```
Model successfully exported to /root/MindBridge/output/mindbridge-qwen3-14b-ft-f16.gguf
```

进度条有时会长时间不刷新，可以另开一个 Terminal 看文件大小是否增长：



```
watch -n 10 'date; ls -lh /root/MindBridge/output/mindbridge-qwen3-14b-ft-f16.gguf; ps aux | grep convert\_hf\_to\_gguf | grep -v grep'
```

### 空间清理（推荐）

F16 GGUF 生成后，HF 格式模型不再需要：



```
rm -rf /root/MindBridge/output/mindbridge-qwen3-14b-ft-hf

df -h /
```

## 11. 量化为 Q4\_K\_M

F16 GGUF 大约 28GB，不适合本地 Mac 使用，需要量化成 4-bit。

如果缺少 cmake：



```
apt update

apt install -y cmake build-essential
```

编译量化工具：



```
cd /root/llama.cpp

cmake -B build

cmake --build build --config Release -j
```

执行量化：



```
./build/bin/llama-quantize \\

&#x20; /root/MindBridge/output/mindbridge-qwen3-14b-ft-f16.gguf \\

&#x20; /root/MindBridge/output/mindbridge-qwen3-14b-ft-q4\_k\_m.gguf \\

&#x20; Q4\_K\_M
```

量化约 5-10 分钟，成功时会看到类似：



```
model size = 28000.00 MiB

quant size = 8500.00 MiB
```

确认文件：



```
ls -lh /root/MindBridge/output/mindbridge-qwen3-14b-ft-q4\_k\_m.gguf
```

正常大小约 8.5-9.5GB。

量化成功后，删除 F16 文件节省空间：



```
rm -f /root/MindBridge/output/mindbridge-qwen3-14b-ft-f16.gguf

ls -lh /root/MindBridge/output/
```

## 12. 下载最终模型到本地

最终需要下载：



```
/root/MindBridge/output/mindbridge-qwen3-14b-ft-q4\_k\_m.gguf
```

方式 A：scp 下载（推荐），在本地 Mac 终端执行：



```
mkdir -p /Users/lqf/projects/agentcode/mindbridge-py/models/mindbridge-qwen3-14b-ft

scp -P 端口号 root@服务器IP:/root/MindBridge/output/mindbridge-qwen3-14b-ft-q4\_k\_m.gguf \\

&#x20; /Users/lqf/projects/agentcode/mindbridge-py/models/mindbridge-qwen3-14b-ft/
```

方式 B：JupyterLab 下载，在左侧文件浏览器找到该文件，右键 Download。

下载到本地项目后，确认：



```
ls -lh /Users/lqf/projects/agentcode/mindbridge-py/models/mindbridge-qwen3-14b-ft/
```

应看到约 9GB 的文件。如果大小明显不对（如只有几 MB），说明下载不完整，重新下载。

## 13. 本地 Ollama 接入

### 13.1 创建 Modelfile

确保本地 `models/mindbridge-qwen3-14b-ft/Modelfile` 内容如下：



```
FROM ./mindbridge-qwen3-14b-ft-q4\_k\_m.gguf

SYSTEM """

你是 MindBridge 校园心理关怀智能体，由 Qwen3-14B 面向校园心理陪伴场景适配而来。

你需要保持温和、稳定、非评判的表达风格，优先保护学生安全。

当学生表达普通问题时，正常回答，不强行心理测评。

当学生表达情绪困扰时，先共情，再给出具体、可执行的小步骤。

当学生表达自伤、自杀、伤人等高风险信号时，不提供危险细节，鼓励其立即联系身边可信任的人、学校辅导员/心理中心或当地紧急救助。

不要向学生输出后台风险等级、心理报告、评分或诊断结论。

"""

PARAMETER temperature 0.65

PARAMETER top\_p 0.9

PARAMETER repeat\_penalty 1.1

PARAMETER num\_ctx 4096
```

### 13.2 创建本地 Ollama 模型

确保本地 Ollama 已启动（菜单栏有 Ollama 图标），然后执行：



```
cd /Users/lqf/projects/agentcode/mindbridge-py/models/mindbridge-qwen3-14b-ft

ollama create mindbridge-qwen3-14b-ft -f Modelfile
```

检查：



```
ollama list
```

应看到：



```
NAME                            ID              SIZE    MODIFIED

mindbridge-qwen3-14b-ft:latest  xxxxxxxxxxxx    9.0 GB  2 minutes ago
```

### 13.3 测试对话



```
ollama run mindbridge-qwen3-14b-ft
```

输入测试问题：



```
最近考试快到了，我总担心自己复习不完，晚上也睡不踏实。
```

模型应该以共情、温和的语气回复。输入 /bye 退出。

14B Q4 模型约占用 9-11GB 内存，24G Mac Air 可以轻松运行，推理速度约 8-15 token/s。

## 14. 项目中如何确认调用的是微调模型

如果项目代码中写死了旧模型名，需要把 `mindbridge-qwen2.5-7b-ft` 改为 `mindbridge-qwen3-14b-ft`。检查以下文件：



```
grep -r "mindbridge-qwen2.5-7b" /Users/lqf/projects/agentcode/mindbridge-py --include="\*.yml" --include="\*.yaml" --include="\*.sh" --include="\*.py" -l
```

通常需要修改：



* application.yml 或 config.py 中的 ollama.model

* scripts/run-dev.sh 中的 OLLAMA\_MODEL

* 前端代码中的模型名（如果有）

整体链路：



```
application.yml / run-dev.sh

&#x20;      ↓

mindbridge-qwen3-14b-ft:latest

&#x20;      ↓

OllamaAiClient

&#x20;      ↓

Ollama

&#x20;      ↓

Modelfile

&#x20;      ↓

mindbridge-qwen3-14b-ft-q4\_k\_m.gguf
```

## 15. 模型评估（推荐）

训练完成后，用 test.json 和 campus\_eval.json 做客观评估，这两个集训练时完全不碰。



* test.json（6454 条）：通用评估，衡量模型在所有话题上的整体效果

* campus\_eval.json（3180 条）：校园场景专项评估，专门衡量考试焦虑、宿舍关系、失恋、就业压力等学生高频场景

评估方式：把评估集中的前几轮对话输入模型，让模型生成回复，和数据集中的标准回复对比，或人工打分。也可以用同样的输入调用通用 API（如 Doubao、GPT-4），从以下维度对比：



| 维度     | 说明             |
| ------ | -------------- |
| 共情能力   | 是否先共情、再建议      |
| 上下文连贯性 | 是否记住前文内容       |
| 回复自然度  | 是否像真人咨询师，而非模板化 |
| 安全性    | 高风险场景是否正确转介    |
| 回复长度   | 是否适中（不过短也不过长）  |

## 16. 常见问题

### 16.1 训练 OOM（Out of Memory）

48G 显存一般不会 OOM。如果报错：



1. 降低 --per\_device\_train\_batch\_size 从 2 到 1

2. 降低 --max\_length 从 2048 到 1024

3. 检查是否有其他进程占用 GPU：nvidia-smi

### 16.2 磁盘满（No space left on device）

100G 系统盘通常不会满（峰值约 87GB），如果同时保留了所有中间文件也可能遇到：



```
\# 检查空间

df -h

\# 查看大文件

du -sh /root/\* 2>/dev/null | sort -rh | head -10

du -sh /root/.cache/\* 2>/dev/null | sort -rh | head -10

\# 清理基座模型缓存（训练完成后）

rm -rf /root/.cache/modelscope/hub/models/Qwen/

rm -rf /root/.cache/huggingface/hub/

\# 清理 pip 缓存

rm -rf /root/.cache/pip
```

### 16.3 训练中断后恢复

如果训练意外中断（网络断开、服务器重启），可以从最后一个 checkpoint 恢复：



```
\# 找到最后一个 checkpoint

ls -lt /root/MindBridge/output/qwen3-14b-mindbridge-lora/\*/checkpoint-\* | head -5

\# 在训练命令中加上 --resume\_from\_checkpoint

\--resume\_from\_checkpoint /root/MindBridge/output/qwen3-14b-mindbridge-lora/v1-xxx/checkpoint-10000
```

用 nohup 后台运行可以避免 SSH 断开导致训练中断。

### 16.4 GGUF 转换一直停在 0% 或 7%

先检查文件大小是否增长：



```
ls -lh /root/MindBridge/output/mindbridge-qwen3-14b-ft-f16.gguf

ps aux | grep convert\_hf\_to\_gguf | grep -v grep
```

如果文件在增长，说明还在写。云服务器系统盘 IO 慢时，进度条可能长时间不刷新，14B 模型转换约 10-20 分钟。

### 16.5 llama-quantize: failed ... iostream error

大概率是磁盘空间不足。检查：



```
df -h
```

清理不需要的文件后重新量化。

### 16.6 bash: cmake: 未找到命令



```
apt update

apt install -y cmake build-essential
```

或：



```
pip install cmake
```

### 16.7 ms-swift 版本不兼容 Qwen3

如果报错模型不支持，升级 ms-swift：



```
pip install -U ms-swift \\

&#x20; -i https://mirrors.aliyun.com/pypi/simple \\

&#x20; \--trusted-host mirrors.aliyun.com
```

### 16.8 模型下载慢

Qwen3-14B 基座模型 28GB，如果 ms-swift 自动下载慢，可以手动用 modelscope CLI 下载：



```
pip install modelscope

modelscope download --model Qwen/Qwen3-14B-Instruct --local\_dir /root/.cache/modelscope/hub/models/Qwen/Qwen3-14B-Instruct
```

### 16.9 本地 Ollama 推理慢

14B Q4 在 Mac Air 24G 上推理速度约 8-15 token/s。如果明显更慢：



```
\# 检查是否有其他进程占用内存

ps aux --sort=-%mem | head -10

\# 检查 Ollama 是否使用了 Metal 加速

ollama ps
```

Mac Air 无风扇，长时间推理会降频。如果需要更快速度，可以考虑 Q3\_K\_M 量化（更小更快，但质量略降）。

### 16.10 ValueError: remaining\_argv: \['--train\_type', 'lora']

新版 ms-swift 使用 `--tuner_type lora`，不要使用 `--train_type lora`。

## 17. 云服务器释放

确认以下文件已经下载到本地：



```
models/mindbridge-qwen3-14b-ft/mindbridge-qwen3-14b-ft-q4\_k\_m.gguf
```

并确认本地 Ollama 已经创建：



```
ollama list | grep mindbridge-qwen3-14b-ft
```

如果需要保留训练日志，先下载：



```
scp -P 端口号 root@服务器IP:/root/MindBridge/train.log ./
```

然后在 AI Gate 控制台释放实例。注意：如果云平台有「关机」和「释放」，应选择「释放」，避免继续计费。释放后服务器上所有数据都会被删除。

## 18. 产物清单

云端训练产物（释放后删除）：



```
/root/MindBridge/output/qwen3-14b-mindbridge-lora/.../checkpoint-27590/  （LoRA adapter）

/root/MindBridge/output/mindbridge-qwen3-14b-ft-hf/                        （合并后 HF 模型）

/root/MindBridge/output/mindbridge-qwen3-14b-ft-f16.gguf                   （F16 GGUF）

/root/MindBridge/output/mindbridge-qwen3-14b-ft-q4\_k\_m.gguf                （Q4 GGUF）

/root/MindBridge/train.log                                                  （训练日志）
```

最终交付 / 运行只需要：



```
models/mindbridge-qwen3-14b-ft/mindbridge-qwen3-14b-ft-q4\_k\_m.gguf

models/mindbridge-qwen3-14b-ft/Modelfile
```

如果需要证明训练过程，可以额外保留：



```
checkpoint-27590/

logging.jsonl

args.json
```



***

*文档版本：v2.0 | 最后更新：2026-09-01*

*适用模型：Qwen3-14B-Instruct | 数据集：SoulChatCorpus | 框架：ms-swift*