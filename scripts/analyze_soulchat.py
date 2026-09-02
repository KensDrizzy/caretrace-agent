# -*- coding: utf-8 -*-
"""SoulChatCorpus 数据集结构分析脚本
用法: python3 analyze_soulchat.py <json路径>
输出: 样本数、轮数分布、话题分布、消息长度统计、学生场景关键词覆盖
"""
import json
import sys
import re
from collections import Counter, defaultdict

def main(path):
    print(f"加载文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"总对话数: {len(data)}")

    # 轮数与消息长度
    turn_counts = []
    user_chars, assistant_chars = [], []
    total_user_turns, total_assistant_turns = 0, 0
    topics = Counter()
    # 学生/校园场景关键词
    kw = ["考试", "学业", "学习", "考研", "考公", "就业", "实习", "求职", "宿舍",
          "同学", "室友", "室友关系", "导师", "老师", "课程", "毕业", "论文",
          "高考", "上课", "大学", "校园", "失恋", "恋爱", "表白", "成绩", "挂科"]
    kw_counter = Counter()
    samples_with_kw = 0
    empty_turn = 0
    only_user = 0
    no_assistant = 0
    bad_roles = 0

    for s in data:
        msgs = s.get("messages") or []
        turn_counts.append(len(msgs))
        t = s.get("topic") or "未知"
        topics[t] += 1
        joined = ""
        for m in msgs:
            role = m.get("role")
            content = m.get("content") or ""
            joined += content
            if role == "user":
                total_user_turns += 1
                user_chars.append(len(content))
            elif role == "assistant":
                total_assistant_turns += 1
                assistant_chars.append(len(content))
            else:
                bad_roles += 1
        if len(msgs) == 0:
            empty_turn += 1
        if not any(m.get("role") == "assistant" for m in msgs):
            no_assistant += 1
        if sum(1 for m in msgs if m.get("role") == "user") > 0 and not any(m.get("role") == "assistant" for m in msgs):
            only_user += 1
        hit = [k for k in kw if k in joined]
        if hit:
            samples_with_kw += 1
            for k in hit:
                kw_counter[k] += 1

    def pct(x, n):
        return f"{x/n*100:.2f}%"

    print(f"\n总轮次(user+assistant): {total_user_turns + total_assistant_turns}")
    print(f"  user轮: {total_user_turns}  assistant轮: {total_assistant_turns}")
    print(f"轮次统计(每条样本的消息数): "
          f"min={min(turn_counts)} max={max(turn_counts)} "
          f"avg={sum(turn_counts)/len(turn_counts):.2f} "
          f"median={sorted(turn_counts)[len(turn_counts)//2]}")
    print(f"\n轮数分布: ")
    dist = Counter()
    for t in turn_counts:
        if t <= 2:
            dist["1-2轮"] += 1
        elif t <= 4:
            dist["3-4轮"] += 1
        elif t <= 8:
            dist["5-8轮"] += 1
        elif t <= 16:
            dist["9-16轮"] += 1
        else:
            dist["17轮以上"] += 1
    for k in ["1-2轮", "3-4轮", "5-8轮", "9-16轮", "17轮以上"]:
        c = dist.get(k, 0)
        print(f"  {k:8s}: {c:>8d}  {pct(c, len(data))}")

    if user_chars:
        print(f"\nuser消息长度(字符): avg={sum(user_chars)/len(user_chars):.1f} max={max(user_chars)}")
    if assistant_chars:
        print(f"assistant消息长度(字符): avg={sum(assistant_chars)/len(assistant_chars):.1f} max={max(assistant_chars)}")

    print(f"\n话题分布(Top 25):")
    for t, c in topics.most_common(25):
        print(f"  {t}: {c}  {pct(c, len(data))}")

    print(f"\n学生/校园场景关键词命中: {samples_with_kw} 样本 ({pct(samples_with_kw, len(data))})")
    for k, c in kw_counter.most_common(30):
        print(f"  {k}: {c}")

    print(f"\n异常检查: 空消息样本={empty_turn}, 无assistant回复样本={no_assistant}, 非法role数={bad_roles}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "SoulChatCorpus-sft-multi-Turn.json")
