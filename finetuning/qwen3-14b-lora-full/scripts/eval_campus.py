import json
import requests
import random
import time
from pathlib import Path

# 配置
MODEL_NAME = "caretrace-qwen3-14b-ft"
OLLAMA_URL = "http://localhost:11434/api/chat"
EVAL_FILE = "/Users/lqf/projects/agentcode/mindbridge-py/data/dataset/campus_eval.json"
OUTPUT_FILE = "/Users/lqf/projects/agentcode/mindbridge-py/finetuning/qwen3-14b-lora-full/eval_results.json"
SAMPLE_SIZE = 50  # 先抽50条评估，3180条全跑太慢
CONTEXT_ROUNDS = 2  # 用前2轮对话做上下文，第3轮user的话做输入
SEED = 42

def generate_reply(messages):
    """调用 Ollama 生成回复"""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.65,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": 512
        }
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception as e:
        return f"[ERROR] {e}"

def main():
    random.seed(SEED)

    # 读取评估集
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"评估集总条数: {len(data)}")

    # 筛选至少有 CONTEXT_ROUNDS*2+1 条消息的对话（足够做上下文+输入）
    min_msgs = CONTEXT_ROUNDS * 2 + 1
    valid = [d for d in data if len(d["messages"]) >= min_msgs]
    print(f"符合长度要求的条数: {len(valid)}")

    # 抽样
    samples = random.sample(valid, min(SAMPLE_SIZE, len(valid)))
    print(f"抽样评估条数: {len(samples)}")

    results = []
    for i, item in enumerate(samples):
        msgs = item["messages"]
        # 取前 CONTEXT_ROUNDS*2 条消息做上下文，下一条 user 消息做输入
        context = msgs[:CONTEXT_ROUNDS * 2]
        input_user = msgs[CONTEXT_ROUNDS * 2]
        # 标准回复（GT）
        gt_reply = msgs[CONTEXT_ROUNDS * 2 + 1]["content"] if len(msgs) > CONTEXT_ROUNDS * 2 + 1 else ""

        # 构造模型输入
        model_messages = [{"role": m["role"], "content": m["content"]} for m in context]
        model_messages.append({"role": "user", "content": input_user["content"]})

        print(f"[{i+1}/{len(samples)}] topic={item['topic']}, id={item['id']}...", end=" ", flush=True)
        start = time.time()
        model_reply = generate_reply(model_messages)
        elapsed = time.time() - start
        print(f"done ({elapsed:.1f}s)")

        results.append({
            "id": item["id"],
            "topic": item["topic"],
            "context": [{"role": m["role"], "content": m["content"]} for m in context],
            "input": input_user["content"],
            "gt_reply": gt_reply,
            "model_reply": model_reply,
            "elapsed_seconds": round(elapsed, 1)
        })

        # 每10条保存一次，防止中断丢失
        if (i + 1) % 10 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # 最终保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n评估完成，结果保存到: {OUTPUT_FILE}")
    print(f"共评估 {len(results)} 条，平均耗时 {sum(r['elapsed_seconds'] for r in results)/len(results):.1f}s/条")

if __name__ == "__main__":
    main()
