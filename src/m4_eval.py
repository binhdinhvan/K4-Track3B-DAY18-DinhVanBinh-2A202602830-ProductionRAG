from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    TEST_SET_PATH,
)


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    empty_result = {name: 0.0 for name in metric_names}
    empty_result["per_question"] = []

    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("Evaluation inputs must have the same length")
    if not questions:
        return empty_result

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from langchain_openai import ChatOpenAI
        from ragas.embeddings import HuggingfaceEmbeddings

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not configured for RAGAS")

        evaluator_llm = ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0.0,
        )
        evaluator_embeddings = HuggingfaceEmbeddings(
            model_name=EMBEDDING_MODEL,
        )
        evaluation = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            raise_exceptions=True,
        )
        frame = evaluation.to_pandas()
        if len(frame) != len(questions) or frame[metric_names].isna().any().any():
            raise RuntimeError("RAGAS returned incomplete metrics; do not treat missing scores as zero")

        def score(row, name):
            value = row.get(name, 0.0)
            return 0.0 if value != value else float(value)

        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                faithfulness=score(row, "faithfulness"),
                answer_relevancy=score(row, "answer_relevancy"),
                context_precision=score(row, "context_precision"),
                context_recall=score(row, "context_recall"),
            )
            for _, row in frame.iterrows()
        ]
        return {
            **{
                name: (
                    sum(getattr(item, name) for item in per_question)
                    / len(per_question)
                    if per_question else 0.0
                )
                for name in metric_names
            },
            "per_question": per_question,
        }
    except Exception as exc:
        print(f"  ⚠️  RAGAS evaluation failed: {exc}")
        empty_result["evaluation_error"] = str(exc)
        return empty_result


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []

    diagnostic_tree = {
        "faithfulness": (
            "LLM hallucinating",
            "Tighten the prompt and lower the generation temperature",
        ),
        "context_recall": (
            "Missing relevant chunks",
            "Improve chunking or add BM25 retrieval",
        ),
        "context_precision": (
            "Too many irrelevant chunks",
            "Add reranking or metadata filtering",
        ),
        "answer_relevancy": (
            "Answer does not match the question",
            "Improve the answer prompt template",
        ),
    }
    metric_names = tuple(diagnostic_tree)
    analyzed = []
    for result in eval_results:
        scores = {name: getattr(result, name) for name in metric_names}
        average = sum(scores.values()) / len(scores)
        worst_metric = min(metric_names, key=scores.__getitem__)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": result.question,
            "answer": result.answer,
            "ground_truth": result.ground_truth,
            "worst_metric": worst_metric,
            "score": scores[worst_metric],
            "average_score": average,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    analyzed.sort(key=lambda item: item["average_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
        "per_question": [vars(item) for item in results.get("per_question", [])],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
