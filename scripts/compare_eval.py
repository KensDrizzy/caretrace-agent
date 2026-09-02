#!/usr/bin/env python3
"""
对比微调模型 vs 未微调基座模型的评估结果。
"""
import json
import re
from pathlib import Path
from collections import Counter

FT_FILE = "/Users/lqf/projects/agentcode/mindbridge-py/finetuning/qwen3-14b-lora-full/eval_results.json"
BASE_FILE = "/Users/lqf/projects/agentcode/mindbridge-py/finetuning/qwen3-14b-lora-full/eval_results_base.json"
REPORT_FILE = "/Users/lqf/projects/agentcode/mindbridge-py/finetuning/qwen3-14b-lora-full/eval_comparison.md"

EMPATHY_KEYWORDS = [
    "理解", "感受", "感受到", "别担心", "不要担心", "没关系", "很正常",
    "不容易", "辛苦", "难过", "痛苦", "焦虑", "压力", "陪伴", "支持",
    "抱抱", "共情", "倾听", "我在", "懂你", "明白", "体会", "心疼"
]

TEMPLATE_PATTERNS = [
    r"^你好", r"^您好", r"^听到你说", r"^我理解", r"^我能理解",
    r"^感受到", r"^这听起来", r"^这是很正常", r"^很多人"
]

def char_jaccard(s1, s2):
    set1, set2 = set(s1), set(s2)
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0.0

def count_empathy(text):
    return sum(1 for kw in EMPATHY_KEYWORDS if kw in text)

def is_template_start(text):
    return any(re.match(p, text.strip()) for p in TEMPLATE_PATTERNS)

def calc_metrics(data):
    valid = [d for d in data if not d["model_reply"].startswith("[ERROR]")]
    times = sorted([d["elapsed_seconds"] for d in valid])
    model_lens = [len(d["model_reply"]) for d in valid]
    gt_lens = [len(d["gt_reply"]) for d in valid]
    sims = [char_jaccard(d["model_reply"], d["gt_reply"]) for d in valid]
    emp = [count_empathy(d["model_reply"]) for d in valid]
    gt_emp = [count_empathy(d["gt_reply"]) for d in valid]
    no_emp = sum(1 for c in emp if c == 0)
    template = sum(1 for d in valid if is_template_start(d["model_reply"]))
    unique = len(set(d["model_reply"] for d in valid))

    return {
        "total": len(data),
        "valid": len(valid),
        "errors": len(data) - len(valid),
        "avg_time": sum(times) / len(times) if times else 0,
        "median_time": times[len(times)//2] if times else 0,
        "avg_model_len": sum(model_lens) / len(model_lens) if model_lens else 0,
        "avg_gt_len": sum(gt_lens) / len(gt_lens) if gt_lens else 0,
        "avg_sim": sum(sims) / len(sims) if sims else 0,
        "avg_empathy": sum(emp) / len(emp) if emp else 0,
        "avg_gt_empathy": sum(gt_emp) / len(gt_emp) if gt_emp else 0,
        "no_empathy_ratio": no_emp / len(valid) * 100 if valid else 0,
        "template_ratio": template / len(valid) * 100 if valid else 0,
        "diversity": unique / len(valid) * 100 if valid else 0,
    }

def main():
    with open(FT_FILE, "r", encoding="utf-8") as f:
        ft_data = json.load(f)
    with open(BASE_FILE, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    ft = calc_metrics(ft_data)
    base = calc_metrics(base_data)

    lines = []
    lines.append("# 微调模型 vs 未微调基座模型 评估对比报告")
    lines.append("")
    lines.append(f"评估样本：{ft['total']} 条（完全相同的输入， campus_eval.json 抽样）")
    lines.append(f"微调模型：caretrace-qwen3-14b-ft（SoulChatCorpus LoRA 微调）")
    lines.append(f"基座模型：qwen3:14b（未微调，Ollama 官方 Q4）")
    lines.append("")

    lines.append("## 一、核心指标对比")
    lines.append("")
    lines.append("| 指标 | 微调模型 | 基座模型 | 差异 | 微调是否更好 |")
    lines.append("|---|---|---|---|---|")

    def diff_row(name, ft_val, base_val, unit="", higher_better=True, fmt="{:.1f}"):
        diff = ft_val - base_val
        if higher_better:
            better = "✅ 是" if diff > 0 else ("❌ 否" if diff < 0 else "➖ 持平")
        else:
            better = "✅ 是" if diff < 0 else ("❌ 否" if diff > 0 else "➖ 持平")
        sign = "+" if diff > 0 else ""
        return f"| {name} | {fmt.format(ft_val)}{unit} | {fmt.format(base_val)}{unit} | {sign}{fmt.format(diff)}{unit} | {better} |"

    lines.append(diff_row("生成成功率", ft["valid"]/ft["total"]*100, base["valid"]/base["total"]*100, "%", True, "{:.0f}"))
    lines.append(diff_row("平均生成耗时", ft["avg_time"], base["avg_time"], "s", False))
    lines.append(diff_row("平均回复长度", ft["avg_model_len"], base["avg_model_len"], "字", True, "{:.0f}"))
    lines.append(diff_row("与GT相似度(Jaccard)", ft["avg_sim"], base["avg_sim"], "", True, "{:.3f}"))
    lines.append(diff_row("共情关键词命中", ft["avg_empathy"], base["avg_empathy"], "个/条", True))
    lines.append(diff_row("无共情回复占比", ft["no_empathy_ratio"], base["no_empathy_ratio"], "%", False))
    lines.append(diff_row("模板化开头占比", ft["template_ratio"], base["template_ratio"], "%", False))
    lines.append(diff_row("回复多样性", ft["diversity"], base["diversity"], "%", True))
    lines.append("")

    lines.append("## 二、逐条对比（前10条预览）")
    lines.append("")
    lines.append("> 完整对比数据见 eval_results.json（微调）和 eval_results_base.json（基座）。")
    lines.append("")

    for i in range(min(10, len(ft_data))):
        ft_item = ft_data[i]
        base_item = base_data[i]
        lines.append(f"### 第 {i+1} 条（id={ft_item['id']}, topic={ft_item['topic']}）")
        lines.append("")
        lines.append(f"**输入：** {ft_item['input'][:100]}{'...' if len(ft_item['input'])>100 else ''}")
        lines.append("")
        lines.append(f"**标准答案(GT)：** {ft_item['gt_reply']}")
        lines.append("")
        lines.append(f"**微调模型：** {ft_item['model_reply']}")
        lines.append("")
        lines.append(f"**基座模型：** {base_item['model_reply']}")
        lines.append("")
        ft_emp = count_empathy(ft_item["model_reply"])
        base_emp = count_empathy(base_item["model_reply"])
        lines.append(f"*共情关键词：微调 {ft_emp} 个 vs 基座 {base_emp} 个 | 耗时：微调 {ft_item['elapsed_seconds']}s vs 基座 {base_item['elapsed_seconds']}s*")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## 三、结论与建议")
    lines.append("")
    lines.append("### 自动指标层面")
    lines.append("")
    # 统计微调更好的指标数
    better_count = 0
    total_compare = 0
    comparisons = [
        (ft["avg_empathy"] > base["avg_empathy"], "共情关键词"),
        (ft["no_empathy_ratio"] < base["no_empathy_ratio"], "无共情占比(低好)"),
        (ft["template_ratio"] < base["template_ratio"], "模板化占比(低好)"),
        (ft["diversity"] >= base["diversity"], "回复多样性"),
        (ft["avg_model_len"] > base["avg_model_len"], "回复长度"),
    ]
    for is_better, name in comparisons:
        total_compare += 1
        if is_better:
            better_count += 1
    lines.append(f"在 {total_compare} 项核心质量指标中，微调模型在 {better_count} 项上优于基座模型。")
    lines.append("")
    lines.append("### 人工判断建议")
    lines.append("")
    lines.append("自动指标只能反映部分情况，建议重点看第二节的逐条对比，从以下角度判断：")
    lines.append("")
    lines.append("1. **共情真诚度**：微调模型的共情是真的在接纳情绪，还是只是堆了关键词？")
    lines.append("2. **建议具体性**：微调模型给的建议是否更具体、更有针对性？")
    lines.append("3. **回复自然度**：哪个模型的回复更像真人咨询师说话？")
    lines.append("4. **上下文连贯**：哪个模型更能记住前2轮对话的内容？")
    lines.append("5. **风格一致性**：微调模型是否形成了稳定的咨询风格？")
    lines.append("")
    lines.append("### 如果微调提升不明显")
    lines.append("")
    lines.append("- 可能是训练数据量不够或质量不够高")
    lines.append("- 可以尝试增加训练 epochs 或调整学习率")
    lines.append("- 可以加入更多真实咨询记录（当前数据多为 GPT-4 合成）")
    lines.append("- 可以尝试更高 rank 的 LoRA（如 rank=32 或 64）")
    lines.append("")

    report = "\n".join(lines)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    # 终端打印核心对比
    print("=" * 70)
    print("微调模型 vs 基座模型 核心指标对比")
    print("=" * 70)
    print(f"{'指标':<20} {'微调模型':>12} {'基座模型':>12} {'差异':>10}")
    print("-" * 70)
    print(f"{'生成成功率':<20} {ft['valid']/ft['total']*100:>11.0f}% {base['valid']/base['total']*100:>11.0f}% {ft['valid']-base['valid']:>+9d}")
    print(f"{'平均耗时':<20} {ft['avg_time']:>11.1f}s {base['avg_time']:>11.1f}s {ft['avg_time']-base['avg_time']:>+9.1f}s")
    print(f"{'平均回复长度':<20} {ft['avg_model_len']:>11.0f}字 {base['avg_model_len']:>11.0f}字 {ft['avg_model_len']-base['avg_model_len']:>+9.0f}字")
    print(f"{'与GT相似度':<20} {ft['avg_sim']:>12.3f} {base['avg_sim']:>12.3f} {ft['avg_sim']-base['avg_sim']:>+10.3f}")
    print(f"{'共情关键词/条':<20} {ft['avg_empathy']:>12.1f} {base['avg_empathy']:>12.1f} {ft['avg_empathy']-base['avg_empathy']:>+10.1f}")
    print(f"{'无共情回复占比':<20} {ft['no_empathy_ratio']:>11.1f}% {base['no_empathy_ratio']:>11.1f}% {ft['no_empathy_ratio']-base['no_empathy_ratio']:>+9.1f}%")
    print(f"{'模板化开头占比':<20} {ft['template_ratio']:>11.1f}% {base['template_ratio']:>11.1f}% {ft['template_ratio']-base['template_ratio']:>+9.1f}%")
    print(f"{'回复多样性':<20} {ft['diversity']:>11.1f}% {base['diversity']:>11.1f}% {ft['diversity']-base['diversity']:>+9.1f}%")
    print("=" * 70)
    print(f"详细对比报告已保存到: {REPORT_FILE}")

if __name__ == "__main__":
    main()
