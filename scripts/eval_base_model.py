#!/usr/bin/env python3
"""
用已有的评估样本（eval_results.json）评估另一个模型（如未微调的基座模型）。
保证输入完全一致，对比才公平。
"""
import json
import requests
import time
from pathlib import Path

# 配置
MODEL_NAME = "qwen3:14b"  # 未微调的基座模型
OLLAMA_URL = "http://localhost:11434/api/chat"
INPUT_FILE = "/Users/lqf/projects/agentcode/mindbridge-py/finetuning/qwen3-14b-lora-full/eval_results.json"
OUTPUT_FILE = "/Users/lqf/projects/agentcode/mindbridge-py/finetuning/qwen3-14b-lora-full/eval_results_base.json"

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
    # 读取微调模型的评估结果，复用完全相同的样本
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        ft_results = json.load(f)

    print(f"复用 {len(ft_results)} 条评估样本")
    print(f"评估模型: {MODEL_NAME}")
    print("=" * 60)

    results = []
    for i, item in enumerate(ft_results):
        # 直接复用 context + input，保证输入完全一致
        model_messages = [{"role": m["role"], "content": m["content"]} for m in item["context"]]
        model_messages.append({"role": "user", "content": item["input"]})

        print(f"[{i+1}/{len(ft_results)}] topic={item['topic']}, id={item['id']}...", end=" ", flush=True)
        start = time.time()
        model_reply = generate_reply(model_messages)
        elapsed = time.time() - start
        print(f"done ({elapsed:.1f}s)")

        results.append({
            "id": item["id"],
            "topic": item["topic"],
            "context": item["context"],
            "input": item["input"],
            "gt_reply": item["gt_reply"],
            "model_reply": model_reply,
            "elapsed_seconds": round(elapsed, 1)
        })

        # 每10条保存一次
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
