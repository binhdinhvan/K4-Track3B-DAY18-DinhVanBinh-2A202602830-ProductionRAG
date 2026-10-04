"""Compare concise answers on all saved contexts without changing retrieval.

Run: python scripts/evaluate_answer_prompt.py
Outputs are local candidate artifacts; the published report is not overwritten.
"""
from pathlib import Path
from datetime import datetime
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import create_openai_client, OPENAI_MODEL
from src.pipeline import generate_grounded_answer
from src.m4_eval import evaluate_ragas, failure_analysis, save_report


def main():
    inputs = json.loads(Path("reports/evaluation_inputs.json").read_text(encoding="utf-8"))
    before = json.loads(Path("reports/ragas_report.json").read_text(encoding="utf-8"))
    if before.get("evaluation_metadata", {}).get("evaluator_model") != OPENAI_MODEL:
        raise RuntimeError("Use the same evaluator model as the previous report for comparison")
    if not (len(inputs["questions"]) == len(inputs["contexts"]) == len(inputs["ground_truths"]) == 20):
        raise RuntimeError("Expected the complete 20-question evaluation set")
    output = Path("reports/history") / ("answer_prompt_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "before.json").write_text(json.dumps(before, ensure_ascii=False, indent=2), encoding="utf-8")
    client = create_openai_client().with_options(timeout=60, max_retries=1)
    answers, timings = [], []
    for index, (question, contexts) in enumerate(zip(inputs["questions"], inputs["contexts"])):
        start = time.perf_counter()
        # Ground truth is not supplied to generation.
        answers.append(generate_grounded_answer(question, contexts, client))
        timings.append((time.perf_counter() - start) * 1000)
        print(f"Generated {index + 1}/{len(inputs['questions'])}: {answers[-1][:120]}", flush=True)
    candidate = {**inputs, "answers": answers}
    (output / "inputs.json").write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
    start = time.perf_counter()
    results = evaluate_ragas(**candidate)
    if len(results.get("per_question", [])) != len(answers):
        raise RuntimeError("Candidate evaluation failed: " + results.get("evaluation_error", "incomplete scores"))
    save_report(results, failure_analysis(results["per_question"], bottom_n=5), str(output / "report.json"))
    metadata = {"generation_ms": timings, "ragas_ms": (time.perf_counter() - start) * 1000,
                "evaluator_model": OPENAI_MODEL, "num_questions": len(answers)}
    (output / "timings.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Before:", before["aggregate"], flush=True)
    print("Candidate:", {k: v for k, v in results.items() if k != "per_question"}, flush=True)


if __name__ == "__main__":
    main()
