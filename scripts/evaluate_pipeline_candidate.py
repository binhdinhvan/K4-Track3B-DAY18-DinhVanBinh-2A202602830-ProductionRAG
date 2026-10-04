"""Evaluate the actual production pipeline without overwriting submission files."""
from pathlib import Path
import json
import sys
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OPENAI_MODEL
from src.pipeline import build_pipeline, run_query
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report


def main():
    output = Path("reports/history") / ("pipeline_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True)
    before = json.loads(Path("reports/ragas_report.json").read_text(encoding="utf-8"))
    (output / "before.json").write_text(json.dumps(before, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Candidate directory:", output, flush=True)
    search, reranker = build_pipeline()
    inputs = {"questions": [], "answers": [], "contexts": [], "ground_truths": []}
    for index, item in enumerate(load_test_set()):
        answer, contexts = run_query(item["question"], search, reranker)
        for key, value in (("questions", item["question"]), ("answers", answer),
                           ("contexts", contexts), ("ground_truths", item["ground_truth"])):
            inputs[key].append(value)
        print(f"[{index + 1}/20] {answer}", flush=True)
        (output / "inputs.json").write_text(json.dumps(inputs, ensure_ascii=False, indent=2), encoding="utf-8")
    start = time.perf_counter()
    results = evaluate_ragas(**inputs)
    if len(results.get("per_question", [])) != 20:
        raise RuntimeError("Incomplete evaluation: " + results.get("evaluation_error", "missing scores"))
    save_report(results, failure_analysis(results["per_question"], bottom_n=5), str(output / "report.json"))
    latency = {"build_ms": search.build_timings, "queries": search.query_timings,
               "ragas_ms": (time.perf_counter() - start) * 1000, "evaluation_status": "complete",
               "measurement_metadata": {"end_to_end_rerun": True, "evaluator_model": OPENAI_MODEL}}
    (output / "latency.json").write_text(json.dumps(latency, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Before:", before["aggregate"], flush=True)
    print("Candidate:", {k: v for k, v in results.items() if k != "per_question"}, flush=True)


if __name__ == "__main__":
    main()
