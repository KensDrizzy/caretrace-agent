#!/usr/bin/env python3
"""
分析 campus_eval 评估结果，计算量化指标并生成报告。
"""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

EVAL_FILE = "/Users/lqf/projects/agentcode/mindbridge-py/finetuning/qwen3-14b-lora-full/eval_results.json"
REPORT_FILE = "/Users/lqf/projects/agentcode/mindbridge-py/finetuning/qwen3-14b-lora-full/eval_report.md"

# 共情/关怀关键词（心理咨询场景常见表达）
EMPATHY_KEYWORDS = [
    "理解", "感受", "感受到", "别担心", "不要担心", "没关系", "很正常",
    "不容易", "辛苦", "难过", "痛苦", "焦虑", "压力", "陪伴", "支持",
    "抱抱", "共情", "倾听", "我在", "懂你", "明白", "体会", "心疼"
]

# 模板化开头（可能表示回复模式化）
TEMPLATE_PATTERNS = [
    r"^你好", r"^您好", r"^听到你说", r"^我理解", r"^我能理解",
    r"^感受到", r"^这听起来", r"^这是很正常", r"^很多人"
]

def char_jaccard(s1, s2):
    """字符级 Jaccard 相似度"""
    set1 = set(s1)
    set2 = set(s2)
    if not set1 and not set2:
        return 1.0
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0.0

def count_empathy(text):
    """统计共情关键词命中数"""
    return sum(1 for kw in EMPATHY_KEYWORDS if kw in text)

def is_template_start(text):
    """判断是否模板化开头"""
    return any(re.match(p, text.strip()) for p in TEMPLATE_PATTERNS)

def main():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    errors = [d for d in data if d["model_reply"].startswith("[ERROR]")]
    valid = [d for d in data if not d["model_reply"].startswith("[ERROR]")]

    # 耗时统计
    times = [d["elapsed_seconds"] for d in valid]
    times.sort()
    avg_time = sum(times) / len(times) if times else 0
    median_time = times[len(times)//2] if times else 0
    max_time = times[-1] if times else 0
    min_time = times[0] if times else 0

    # 回复长度
    model_lens = [len(d["model_reply"]) for d in valid]
    gt_lens = [len(d["gt_reply"]) for d in valid]
    avg_model_len = sum(model_lens) / len(model_lens) if model_lens else 0
    avg_gt_len = sum(gt_lens) / len(gt_lens) if gt_lens else 0

    # 话题分布
    topic_counter = Counter(d["topic"] for d in data)

    # 和 gt_reply 的相似度
    similarities = [char_jaccard(d["model_reply"], d["gt_reply"]) for d in valid]
    avg_sim = sum(similarities) / len(similarities) if similarities else 0

    # 共情关键词
    model_empathy = [count_empathy(d["model_reply"]) for d in valid]
    gt_empathy = [count_empathy(d["gt_reply"]) for d in valid]
    avg_model_empathy = sum(model_empathy) / len(model_empathy) if model_empathy else 0
    avg_gt_empathy = sum(gt_empathy) / len(gt_empathy) if gt_empathy else 0
    model_no_empathy = sum(1 for c in model_empathy if c == 0)
    gt_no_empathy = sum(1 for c in gt_empathy if c == 0)

    # 模板化开头
    template_count = sum(1 for d in valid if is_template_start(d["model_reply"]))
    template_ratio = template_count / len(valid) * 100 if valid else 0

    # 回复多样性：model_reply 去重比例
    unique_replies = len(set(d["model_reply"] for d in valid))
    diversity = unique_replies / len(valid) * 100 if valid else 0

    # 生成报告
    lines = []
    lines.append("# Qwen3-14B SoulChatCorpus 微调模型评估报告")
    lines.append("")
    lines.append(f"评估集：campus_eval.json（校园场景专项评估集）")
    lines.append(f"评估条数：{total} 条（抽样）")
    lines.append(f"成功生成：{len(valid)} 条")
    lines.append(f"生成失败：{len(errors)} 条")
    lines.append("")

    lines.append("## 一、生成速度")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 平均耗时 | {avg_time:.1f} 秒/条 |")
    lines.append(f"| 中位耗时 | {median_time:.1f} 秒/条 |")
    lines.append(f"| 最快 | {min_time:.1f} 秒 |")
    lines.append(f"| 最慢 | {max_time:.1f} 秒 |")
    lines.append("")

    lines.append("## 二、回复长度")
    lines.append("")
    lines.append("| 指标 | 模型回复 | 标准答案(GT) |")
    lines.append("|---|---|---|")
    lines.append(f"| 平均字符数 | {avg_model_len:.0f} | {avg_gt_len:.0f} |")
    lines.append("")

    lines.append("## 三、话题分布")
    lines.append("")
    lines.append("| 话题 | 条数 | 占比 |")
    lines.append("|---|---|---|")
    for topic, count in topic_counter.most_common():
        lines.append(f"| {topic} | {count} | {count/total*100:.1f}% |")
    lines.append("")

    lines.append("## 四、与标准答案相似度（参考指标）")
    lines.append("")
    lines.append("> 注意：对话回复没有唯一标准答案，相似度低不代表回复质量差，只表示模型回复和数据集中 GPT-4 生成的回复字面不同。")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 字符级 Jaccard 平均相似度 | {avg_sim:.3f} |")
    lines.append("")

    lines.append("## 五、共情能力")
    lines.append("")
    lines.append("| 指标 | 模型回复 | 标准答案(GT) |")
    lines.append("|---|---|---|")
    lines.append(f"| 平均共情关键词命中数 | {avg_model_empathy:.1f} | {avg_gt_empathy:.1f} |")
    lines.append(f"| 无共情关键词的回复占比 | {model_no_empathy}/{len(valid)} ({model_no_empathy/len(valid)*100:.1f}%) | {gt_no_empathy}/{len(valid)} ({gt_no_empathy/len(valid)*100:.1f}%) |")
    lines.append("")
    lines.append(f"共情关键词列表：{', '.join(EMPATHY_KEYWORDS[:10])}...")
    lines.append("")

    lines.append("## 六、回复多样性与模板化")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 回复去重率（多样性） | {diversity:.1f}% |")
    lines.append(f"| 模板化开头占比 | {template_count}/{len(valid)} ({template_ratio:.1f}%) |")
    lines.append("")
    lines.append(f"模板化开头模式：你好/您好/听到你说/我理解/感受到/这听起来/很多人...")
    lines.append("")

    lines.append("## 七、人工评估建议")
    lines.append("")
    lines.append("自动指标只能反映部分情况，建议从以下维度人工抽样对比（看 10-20 条）：")
    lines.append("")
    lines.append("| 维度 | 怎么看 | 打分建议 |")
    lines.append("|---|---|---|")
    lines.append("| 共情能力 | 是否先共情再建议，还是直接给建议 | 1-5分 |")
    lines.append("| 上下文连贯 | 是否记住了前2轮对话的内容 | 1-5分 |")
    lines.append("| 回复自然度 | 是否像真人说话，还是模板化/机械 | 1-5分 |")
    lines.append("| 建议质量 | 建议是否具体、可执行、有针对性 | 1-5分 |")
    lines.append("| 回复长度 | 是否适中，不过短也不过长 | 1-5分 |")
    lines.append("| 安全性 | 有没有不当建议或危险内容 | 通过/不通过 |")
    lines.append("")
    lines.append("可以同时用同样的输入调用通用 API（如 Doubao/GPT-4）做三方对比。")
    lines.append("")

    lines.append("## 八、逐条对比数据")
    lines.append("")
    lines.append("> 完整对比数据见 eval_results.json，以下列出前 10 条预览。")
    lines.append("")
    for i, d in enumerate(valid[:10]):
        lines.append(f"### 第 {i+1} 条（id={d['id']}, topic={d['topic']}, 耗时={d['elapsed_seconds']}s）")
        lines.append("")
        lines.append(f"**输入：** {d['input'][:100]}{'...' if len(d['input'])>100 else ''}")
        lines.append("")
        lines.append(f"**标准答案(GT)：** {d['gt_reply']}")
        lines.append("")
        lines.append(f"**模型回复：** {d['model_reply']}")
        lines.append("")
        sim = char_jaccard(d["model_reply"], d["gt_reply"])
        emp = count_empathy(d["model_reply"])
        lines.append(f"*相似度：{sim:.3f} | 共情关键词：{emp} 个*")
        lines.append("")
        lines.append("---")
        lines.append("")

    report = "\n".join(lines)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    # 同时在终端打印关键指标
    print("=" * 60)
    print("评估指标汇总")
    print("=" * 60)
    print(f"总条数: {total}, 成功: {len(valid)}, 失败: {len(errors)}")
    print(f"平均耗时: {avg_time:.1f}s/条 (中位 {median_time:.1f}s, 最快 {min_time:.1f}s, 最慢 {max_time:.1f}s)")
    print(f"平均回复长度: 模型 {avg_model_len:.0f} 字 vs GT {avg_gt_len:.0f} 字")
    print(f"与GT平均相似度(Jaccard): {avg_sim:.3f}")
    print(f"共情关键词: 模型 {avg_model_empathy:.1f} 个 vs GT {avg_gt_empathy:.1f} 个")
    print(f"无共情回复占比: 模型 {model_no_empathy/len(valid)*100:.1f}% vs GT {gt_no_empathy/len(valid)*100:.1f}%")
    print(f"回复多样性: {diversity:.1f}%")
    print(f"模板化开头占比: {template_ratio:.1f}%")
    print(f"话题分布: {dict(topic_counter.most_common())}")
    print("=" * 60)
    print(f"详细报告已保存到: {REPORT_FILE}")

if __name__ == "__main__":
    main()
