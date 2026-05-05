"""Parse golden_dataset.md → data/golden/golden_dataset.jsonl

Usage:
    python scripts/parse_golden_dataset.py
    python scripts/parse_golden_dataset.py --input data/golden/golden_dataset.md
    python scripts/parse_golden_dataset.py --validate   # strict: 报告所有空字段

Output: data/golden/golden_dataset.jsonl
Each line is a JSON object:
    {
        "id":         "Q001",
        "category":   "Radiology",
        "difficulty": "Easy",
        "question":   "What is ...",
        "answer":     "The typical ...",
        "notes":      "..."        # 可能为空字符串
    }
"""

import argparse
import io
import json
import re
import sys
from pathlib import Path

# Windows 终端默认 GBK，强制 UTF-8 输出避免 UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 合法的 Category 值（大小写不敏感）
VALID_CATEGORIES = {
    "pharmacology", "oncology", "radiology", "cardiology",
    "neurology", "infectious disease", "general",
}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def parse_field(block: str, field: str) -> str:
    """从一个题目块中提取单个字段的值（支持多行）。

    格式要求：
        **FieldName**: 第一行内容
        继续的内容（可多行）

    遇到下一个 **Xxx**: 或块结束时停止。
    """
    # 匹配 **Field**: 到下一个 **或块结束
    pattern = rf"\*\*{re.escape(field)}\*\*\s*:(.*?)(?=\n\*\*[A-Za-z]|\Z)"
    m = re.search(pattern, block, re.DOTALL | re.IGNORECASE)
    if not m:
        return ""
    return m.group(1).strip()


def parse_markdown(md_path: Path) -> list[dict]:
    text = md_path.read_text(encoding="utf-8")

    # 按 ## Qxxx 分割，每个题目是一个块
    # 允许 Q001、Q01、Q1 等写法（1-3位数字）
    blocks = re.split(r"\n(?=##\s+Q\d+)", text)

    entries = []
    for block in blocks:
        # 只处理以 ## Qxxx 开头的块
        id_match = re.match(r"##\s+(Q\d+)", block.strip())
        if not id_match:
            continue

        qid = id_match.group(1).upper()  # 统一大写，如 Q001

        entry = {
            "id":         qid,
            "category":   parse_field(block, "Category"),
            "difficulty": parse_field(block, "Difficulty"),
            "question":   parse_field(block, "Question"),
            "answer":     parse_field(block, "Answer"),
            "notes":      parse_field(block, "Notes"),
        }
        entries.append(entry)

    return entries


def validate(entries: list[dict]) -> list[str]:
    """返回所有问题列表（空字段、非法值等）。"""
    problems = []
    seen_ids = set()

    for e in entries:
        qid = e["id"]

        if qid in seen_ids:
            problems.append(f"{qid}: 重复 ID")
        seen_ids.add(qid)

        if not e["question"]:
            problems.append(f"{qid}: Question 为空")
        if not e["answer"]:
            problems.append(f"{qid}: Answer 为空")
        if not e["category"]:
            problems.append(f"{qid}: Category 为空")
        elif e["category"].lower() not in VALID_CATEGORIES:
            problems.append(
                f"{qid}: Category '{e['category']}' 不在合法列表 {sorted(VALID_CATEGORIES)}"
            )
        if not e["difficulty"]:
            problems.append(f"{qid}: Difficulty 为空")
        elif e["difficulty"].lower() not in VALID_DIFFICULTIES:
            problems.append(
                f"{qid}: Difficulty '{e['difficulty']}' 应为 Easy/Medium/Hard"
            )

    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/golden/golden_dataset.md",
        help="Markdown 输入文件路径",
    )
    parser.add_argument(
        "--output",
        default="data/golden/golden_dataset.jsonl",
        help="JSONL 输出文件路径",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="严格校验：报告所有空字段和非法值，有问题时退出码非零",
    )
    args = parser.parse_args()

    md_path = Path(args.input)
    if not md_path.exists():
        print(f"[error] 找不到输入文件: {md_path}", file=sys.stderr)
        sys.exit(1)

    entries = parse_markdown(md_path)
    print(f"[parse] 解析到 {len(entries)} 道题")

    # 过滤掉完全空白的占位条目（Question 和 Answer 都为空）
    filled = [e for e in entries if e["question"] or e["answer"]]
    skipped = len(entries) - len(filled)
    if skipped:
        print(f"[parse] 跳过 {skipped} 道未填写的占位题目")

    if args.validate:
        problems = validate(filled)
        if problems:
            print(f"\n[validate] 发现 {len(problems)} 个问题：")
            for p in problems:
                print(f"  [!] {p}")
            sys.exit(1)
        else:
            print("[validate] 全部通过 OK")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for e in filled:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"[output] 已写入 {len(filled)} 道题 → {out_path}")

    # 打印简要统计
    if filled:
        from collections import Counter
        cats = Counter(e["category"] for e in filled if e["category"])
        diffs = Counter(e["difficulty"] for e in filled if e["difficulty"])
        print(f"\n分类分布: {dict(cats)}")
        print(f"难度分布: {dict(diffs)}")


if __name__ == "__main__":
    main()
