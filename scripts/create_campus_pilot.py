# -*- coding: utf-8 -*-
"""Create a fixed campus-focused SoulChat pilot split.

Source data is restricted to the existing train.json. The held-out test.json
and campus_eval.json are never read, so they remain untouched for evaluation.
"""
import json
import random
from collections import defaultdict
from pathlib import Path


SEED = 42
SOURCE = Path("data/dataset/train.json")
OUT_TRAIN = Path("data/dataset/pilot_train_27k.json")
OUT_VAL = Path("data/dataset/pilot_val_3k.json")

TRAIN_CAMPUS = 21_000
TRAIN_GENERAL = 6_000
VAL_CAMPUS = 2_000
VAL_GENERAL = 1_000

CAMPUS_KW = [
    "考试", "学业", "学习", "考研", "考公", "就业", "实习", "求职", "宿舍",
    "同学", "室友", "导师", "老师", "课程", "毕业", "论文", "高考", "上课",
    "大学", "校园", "失恋", "恋爱", "表白", "成绩", "挂科", "奖学金", "升学",
]


def text_of(sample):
    return "".join((m.get("content") or "") for m in (sample.get("messages") or []))


def is_campus(sample):
    text = text_of(sample)
    return any(keyword in text for keyword in CAMPUS_KW)


def proportional_quotas(groups, total):
    """Allocate an exact total by group size using the largest-remainder rule."""
    if total > sum(len(items) for items in groups.values()):
        raise ValueError(f"requested {total}, available {sum(map(len, groups.values()))}")
    keys = list(groups)
    raw = {key: total * len(groups[key]) / sum(map(len, groups.values())) for key in keys}
    quotas = {key: min(len(groups[key]), int(raw[key])) for key in keys}
    remaining = total - sum(quotas.values())
    order = sorted(keys, key=lambda key: raw[key] - int(raw[key]), reverse=True)
    for key in order:
        if remaining == 0:
            break
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            remaining -= 1
    return quotas


def take_stratified(pool, n, rng):
    groups = defaultdict(list)
    for sample in pool:
        groups[sample.get("topic") or "未明确"].append(sample)
    quotas = proportional_quotas(groups, n)
    selected = []
    selected_ids = set()
    for topic, items in groups.items():
        rng.shuffle(items)
        for sample in items[:quotas[topic]]:
            selected.append(sample)
            selected_ids.add(id(sample))
    rng.shuffle(selected)
    return selected, [sample for sample in pool if id(sample) not in selected_ids]


def save(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def main():
    with SOURCE.open(encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(SEED)

    campus = [sample for sample in data if is_campus(sample)]
    general = [sample for sample in data if not is_campus(sample)]
    if len(campus) < TRAIN_CAMPUS + VAL_CAMPUS:
        raise ValueError("not enough campus samples")
    if len(general) < TRAIN_GENERAL + VAL_GENERAL:
        raise ValueError("not enough general samples")

    campus_train, campus_left = take_stratified(campus, TRAIN_CAMPUS, rng)
    campus_val, _ = take_stratified(campus_left, VAL_CAMPUS, rng)
    general_train, general_left = take_stratified(general, TRAIN_GENERAL, rng)
    general_val, _ = take_stratified(general_left, VAL_GENERAL, rng)

    train = campus_train + general_train
    val = campus_val + general_val
    rng.shuffle(train)
    rng.shuffle(val)
    save(OUT_TRAIN, train)
    save(OUT_VAL, val)

    train_ids = {sample.get("id") for sample in train}
    val_ids = {sample.get("id") for sample in val}
    assert len(train) == 27_000 and len(val) == 3_000
    assert not train_ids.intersection(val_ids)
    print(f"source={SOURCE} total={len(data)} campus={len(campus)} general={len(general)}")
    print(f"train={OUT_TRAIN} total={len(train)} campus={len(campus_train)} general={len(general_train)}")
    print(f"val={OUT_VAL} total={len(val)} campus={len(campus_val)} general={len(general_val)}")
    print("seed=42; topic-stratified within campus/general pools")


if __name__ == "__main__":
    main()
