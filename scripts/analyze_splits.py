# -*- coding: utf-8 -*-
"""四个划分子集的对比统计
用法: python3 analyze_splits.py
输出: 每个子集的数量/轮数/话题/长度/校园占比
"""
import json
import sys
from collections import Counter

FILES = ["train.json", "val.json", "test.json", "campus_eval.json"]
CAMPUS_KW = ["考试", "学业", "学习", "考研", "考公", "就业", "实习", "求职", "宿舍",
             "同学", "室友", "导师", "老师", "课程", "毕业", "论文", "高考", "上课",
             "大学", "校园", "失恋", "恋爱", "表白", "成绩", "挂科", "奖学金", "升学"]

def analyze(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n = len(data)
    turns = [len(s["messages"]) for s in data]
    topics = Counter(s.get("topic") or "未明确" for s in data)
    u_len, a_len, full_len = [], [], []
    campus = 0
    for s in data:
        joined = ""
        for m in s["messages"]:
            c = m.get("content") or ""
            joined += c
            if m["role"] == "user":
                u_len.append(len(c))
            else:
                a_len.append(len(c))
        full_len.append(len(joined))
        if any(k in joined for k in CAMPUS_KW):
            campus += 1
    turns_sorted = sorted(turns)
    full_sorted = sorted(full_len)
    return {
        "n": n,
        "avg_turn": sum(turns) / n,
        "med_turn": turns_sorted[n // 2],
        "max_turn": max(turns),
        "min_turn": min(turns),
        "avg_user": sum(u_len) / len(u_len) if u_len else 0,
        "avg_assistant": sum(a_len) / len(a_len) if a_len else 0,
        "avg_full": sum(full_len) / n,
        "p95_full": full_sorted[int(n * 0.95)],
        "max_full": max(full_len),
        "campus_ratio": campus / n * 100,
        "topics": topics,
    }

def fmt_topics(c, k=6):
    return ", ".join(f"{t}({v})" for t, v in c.most_common(k))

def main():
    for f in FILES:
        try:
            r = analyze(f"data/dataset/{f}")
        except FileNotFoundError:
            print(f"{f}: 文件不存在")
            continue
        print(f"===== {f} =====")
        print(f"  样本数: {r['n']}")
        print(f"  轮数: avg={r['avg_turn']:.1f} median={r['med_turn']} min={r['min_turn']} max={r['max_turn']}")
        print(f"  user消息均长: {r['avg_user']:.1f}字 | assistant均长: {r['avg_assistant']:.1f}字")
        print(f"  整条样本字符: avg={r['avg_full']:.0f} p95={r['p95_full']} max={r['max_full']}")
        print(f"  校园场景占比: {r['campus_ratio']:.1f}%")
        print(f"  话题Top6: {fmt_topics(r['topics'])}")
        print()

if __name__ == "__main__":
    main()
