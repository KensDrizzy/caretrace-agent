# -*- coding: utf-8 -*-
"""SoulChatCorpus 数据清洗与划分脚本
用法: python3 split_soulchat.py
输入: data/dataset/SoulChatCorpus-sft-multi-Turn.json
输出: data/dataset/train.json, val.json, test.json, campus_eval.json
划分策略:
  1. 清洗: 过滤空样本、无 assistant 回复样本、超长样本(总字符>8000)
  2. 去重: 按 messages 序列完全去重
  3. 分层抽样: 按 topic 分层, train 95% / val 2.5% / test 2.5%
  4. 校园专项评估集: 从 test 中筛出命中校园/学生场景关键词的样本
"""
import json
import hashlib
import random
from collections import Counter, defaultdict

random.seed(42)

SRC = "data/dataset/SoulChatCorpus-sft-multi-Turn.json"
OUT_DIR = "data/dataset"
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.95, 0.025, 0.025

# 校园/学生场景关键词
CAMPUS_KW = ["考试", "学业", "学习", "考研", "考公", "就业", "实习", "求职", "宿舍",
             "同学", "室友", "导师", "老师", "课程", "毕业", "论文", "高考", "上课",
             "大学", "校园", "失恋", "恋爱", "表白", "成绩", "挂科", "奖学金", "升学"]

MAX_CHARS = 8000  # 超长样本过滤阈值(约6000+ token)

def hit_campus(text):
    return any(k in text for k in CAMPUS_KW)

def main():
    with open(SRC, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = len(data)
    print(f"原始样本: {total}")

    # ---- 清洗 ----
    cleaned = []
    dropped = {"empty": 0, "no_assistant": 0, "too_long": 0}
    lens = []
    for s in data:
        msgs = s.get("messages") or []
        if not msgs:
            dropped["empty"] += 1
            continue
        if not any(m.get("role") == "assistant" for m in msgs):
            dropped["no_assistant"] += 1
            continue
        full_text = "".join(m.get("content") or "" for m in msgs)
        n = len(full_text)
        lens.append(n)
        if n > MAX_CHARS:
            dropped["too_long"] += 1
            continue
        cleaned.append(s)
    print(f"清洗后: {len(cleaned)} (丢弃: {dropped})")
    lens.sort()
    print(f"样本总字符分布: min={lens[0]} p50={lens[len(lens)//2]} "
          f"p95={lens[int(len(lens)*0.95)]} max={lens[-1]}")

    # ---- 去重 ----
    seen = set()
    dedup = []
    dup_cnt = 0
    for s in cleaned:
        h = hashlib.md5(json.dumps(s["messages"], ensure_ascii=False).encode("utf-8")).hexdigest()
        if h in seen:
            dup_cnt += 1
            continue
        seen.add(h)
        dedup.append(s)
    print(f"去重后: {len(dedup)} (删除重复: {dup_cnt})")

    # ---- 按 topic 分层划分 ----
    buckets = defaultdict(list)
    for s in dedup:
        buckets[s.get("topic") or "未明确"].append(s)

    train, val, test = [], [], []
    for topic, items in buckets.items():
        random.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * TEST_RATIO))
        n_val = max(1, round(n * VAL_RATIO))
        n_train = n - n_test - n_val
        if n_train < 0:
            # 极端小话题兜底
            n_train = max(0, n - 2)
            n_val, n_test = 1, 1 if n >= 2 else (1, 0)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    print(f"划分结果: train={len(train)} val={len(val)} test={len(test)} "
          f"({len(train)/len(dedup)*100:.1f}%/{len(val)/len(dedup)*100:.2f}%/{len(test)/len(dedup)*100:.2f}%)")

    # ---- 校园专项评估集(从 test 中筛选) ----
    campus_eval = [s for s in test if hit_campus(
        "".join(m.get("content") or "" for m in s["messages"]))]
    print(f"校园专项评估集(来自test): {len(campus_eval)}")

    def save(name, arr):
        p = f"{OUT_DIR}/{name}"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(arr, f, ensure_ascii=False, indent=1)
        import os
        print(f"已写出: {p} ({os.path.getsize(p)/1024/1024:.1f}MB)")

    save("train.json", train)
    save("val.json", val)
    save("test.json", test)
    save("campus_eval.json", campus_eval)

    # ---- 划分质量检查 ----
    print("\n=== 划分质量检查 ===")
    def topic_dist(arr):
        c = Counter(s.get("topic") or "未明确" for s in arr)
        return c
    d_all = topic_dist(dedup)
    d_train = topic_dist(train)
    d_test = topic_dist(test)
    print("Top 8 话题占比对比 (总体/训练/测试):")
    for t, c in d_all.most_common(8):
        p_all = c / len(dedup) * 100
        p_tr = d_train.get(t, 0) / len(train) * 100
        p_te = d_test.get(t, 0) / len(test) * 100
        print(f"  {t:6s} 总体{p_all:5.2f}% 训练{p_tr:5.2f}% 测试{p_te:5.2f}%")
    campus_ratio_train = sum(1 for s in train if hit_campus("".join(m.get("content") or "" for m in s["messages"]))) / len(train) * 100
    campus_ratio_test = len(campus_eval) / len(test) * 100
    print(f"校园场景占比: 训练集 {campus_ratio_train:.1f}%  测试集 {campus_ratio_test:.1f}%")

if __name__ == "__main__":
    main()
