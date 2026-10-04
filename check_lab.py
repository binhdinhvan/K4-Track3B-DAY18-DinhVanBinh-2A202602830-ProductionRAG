"""
Kiểm tra định dạng bài nộp trước khi submit.
Chạy: python check_lab.py

⚠️ Lỗi định dạng khiến script chấm tự động không chạy → trừ 5 điểm thủ tục.
"""

import json
import os
import sys
import subprocess
import math
import tempfile
import xml.etree.ElementTree as ET

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def check_file(path: str, required: bool = True) -> bool:
    if os.path.exists(path):
        print(f"  ✅ {path}")
        return True
    elif required:
        print(f"  ❌ THIẾU: {path}")
        return False
    else:
        print(f"  ⚠️  Optional: {path}")
        return True


def check_json(path: str, required_keys: list[str]) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        missing = [k for k in required_keys if k not in data]
        if missing:
            print(f"  ❌ {path} thiếu keys: {missing}")
            return False
        metrics = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
        scores = data.get("aggregate", {})
        if data.get("num_questions", 0) <= 0 or any(
            not isinstance(scores.get(name), (int, float))
            or not math.isfinite(scores[name]) or not 0 <= scores[name] <= 1
            for name in metrics
        ):
            print(f"  ❌ {path} — chưa có kết quả đánh giá hợp lệ")
            return False
        print(f"  ✅ {path} — keys và metrics OK")
        return True
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  ❌ {path} — {e}")
        return False


def check_todos() -> int:
    """Count remaining TODO markers in src/."""
    count = 0
    for root, _, files in os.walk("src"):
        for f in files:
            if f.endswith(".py"):
                with open(os.path.join(root, f), encoding="utf-8") as fh:
                    for line in fh:
                        if "# TODO" in line:
                            count += 1
    return count


def run_tests(offline: bool = False) -> tuple[int, int]:
    """Run pytest and return (passed, total)."""
    try:
        with tempfile.TemporaryDirectory() as directory:
            report = os.path.join(directory, "pytest.xml")
            arguments = ["tests/", "-q", "--tb=short", f"--junitxml={report}"]
            command = [sys.executable, "-m", "pytest", *arguments]
            if offline:
                command = [sys.executable, "-c",
                           "import config; config.OPENAI_API_KEY = ''; import pytest; "
                           + f"raise SystemExit(pytest.main({arguments!r}))"]
            result = subprocess.run(
                command,
                capture_output=True, text=True, timeout=900, encoding="utf-8", errors="replace"
            )
            print(result.stdout)
            if not os.path.exists(report):
                print(result.stderr)
                return 0, 0
            cases = ET.parse(report).findall(".//testcase")
            passed = sum(not any(case.find(tag) is not None for tag in ("failure", "error", "skipped")) for case in cases)
            total = len(cases)
            if result.returncode != 0:
                total = max(total, passed + 1)
            return passed, total
    except Exception as e:
        print(f"  ⚠️  pytest error: {e}")
        return 0, 0


def validate():
    print("🔍 Kiểm tra bài nộp Lab 18: Production RAG\n")
    errors = 0

    # 1. Source files
    print("📁 Source code:")
    for f in ["src/m1_chunking.py", "src/m2_search.py", "src/m3_rerank.py",
              "src/m4_eval.py", "src/m5_enrichment.py", "src/pipeline.py"]:
        if not check_file(f):
            errors += 1

    # 2. Reports
    print("\n📊 Reports:")
    if check_file("reports/ragas_report.json"):
        if not check_json("reports/ragas_report.json", ["aggregate", "num_questions"]):
            errors += 1
    else:
        errors += 1
    check_file("reports/naive_baseline_report.json", required=False)

    # 3. Analysis
    print("\n📝 Analysis:")
    if not check_file("analysis/failure_analysis.md"):
        errors += 1
    else:
        with open("analysis/failure_analysis.md", encoding="utf-8") as f:
            content = f.read()
        if "(copy template)" in content or "[Họ và tên]" in content:
            print("  ❌ Failure analysis còn nội dung template")
            errors += 1

    # 4. Individual reflections
    print("\n👤 Individual reflections:")
    reflections = []
    ref_dir = "analysis/reflections"
    if os.path.isdir(ref_dir):
        reflections.extend([f"{ref_dir}/{f}" for f in os.listdir(ref_dir)
                            if f.startswith("reflection_") and f.endswith(".md") and f != "reflection_TEMPLATE.md"])
    if os.path.isdir("analysis"):
        reflections.extend([f"analysis/{f}" for f in os.listdir("analysis")
                            if f.startswith("reflection_") and f.endswith(".md") and f != "reflection_TEMPLATE.md"])

    if reflections:
        for r in set(reflections):
            print(f"  ✅ {r}")
    else:
        print(f"  ⚠️  Chưa có file reflection cá nhân (đặt tại {ref_dir}/reflection_[HọTên].md hoặc analysis/reflection_[HọTên].md)")
        errors += 1

    # 5. TODO count
    print("\n🔧 TODO markers:")
    todo_count = check_todos()
    if todo_count == 0:
        print("  ✅ Không còn TODO nào")
    else:
        print(f"  ⚠️  Còn {todo_count} TODO chưa implement")
        errors += 1

    # 6. Tests
    print("\n🧪 Auto-tests:")
    offline = "--offline" in sys.argv
    if offline:
        print("  Kiểm tra fallback không gọi API; không thay thế đánh giá RAGAS thật.")
    passed, total = run_tests(offline=offline)
    if total > 0:
        pct = passed / total * 100
        print(f"  {'✅' if pct == 100 else '⚠️'} {passed}/{total} tests passed ({pct:.0f}%)")
        if passed != total:
            errors += 1
    else:
        print("  ⚠️  Không chạy được tests")
        errors += 1

    # 7. Summary
    latency_path = "reports/latency_report.json"
    if os.path.exists(latency_path):
        with open(latency_path, encoding="utf-8") as f:
            status = json.load(f).get("evaluation_status", "complete")
        if status != "complete":
            print("  ⚠️  Lần đánh giá mới chưa hoàn tất; báo cáo RAGAS hiện tại thuộc lần chạy trước.")
            errors += 1
    print("\n" + "=" * 50)
    if errors == 0:
        print("🚀 Bài lab sẵn sàng để nộp!")
    else:
        print(f"❌ Có {errors} lỗi. Sửa trước khi nộp.")
    print("=" * 50)
    return errors


if __name__ == "__main__":
    sys.exit(1 if validate() else 0)
