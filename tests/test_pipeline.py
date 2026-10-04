"""Regression checks for parent retrieval and Markdown code fences."""
import pytest

from src.m1_chunking import chunk_hierarchical, chunk_structure_aware
from src.m2_search import SearchResult
from src.m3_rerank import RerankResult
from src.pipeline import run_query


def test_parent_ids_do_not_collide_across_documents():
    _, first = chunk_hierarchical("Same content", metadata={"source": "a.md"})
    _, second = chunk_hierarchical("Same content", metadata={"source": "b.md"})
    assert first[0].parent_id != second[0].parent_id


def test_children_preserve_whole_words_when_they_fit():
    text = "Nhan vien khong duoc tu y xu ly malware tren may tinh."
    _, children = chunk_hierarchical(text, parent_size=100, child_size=15)
    assert " ".join(child.text for child in children).split() == text.split()
    assert all(len(child.text) <= 15 for child in children)


def test_enrichment_preserves_parent_metadata_and_order(monkeypatch):
    import src.m5_enrichment as m5
    monkeypatch.setattr(m5, "_enrich_single_call", lambda text, source: {
        "context": "Context", "metadata": {"parent_id": "wrong", "source": "wrong"}
    })
    chunks = [{"text": text, "metadata": {"source": "policy.md", "parent_id": str(i)}}
              for i, text in enumerate(("first", "second", "third"))]
    result = m5.enrich_chunks(chunks, max_workers=2)
    assert [item.original_text for item in result] == [chunk["text"] for chunk in chunks]
    assert [item.auto_metadata["parent_id"] for item in result] == ["0", "1", "2"]
    assert all(item.auto_metadata["source"] == "policy.md" for item in result)


@pytest.mark.parametrize("fence", ["```", "~~~~"])
def test_structure_keeps_headers_inside_code_fences(fence):
    text = f"# Policy\n\n{fence}\n# code comment\nprint(1)\n{fence}\n\n## Next\n\nContent"
    chunks = chunk_structure_aware(text)
    assert [c.metadata["section"] for c in chunks] == ["# Policy", "## Next"]
    assert f"{fence}\n# code comment\nprint(1)\n{fence}" in chunks[0].text


def test_query_restores_distinct_parents(monkeypatch):
    import config
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    parents = {"a": "Full policy A", "b": "Full policy B", "c": "Full policy C"}
    hits = [SearchResult("fragment", 1.0, {"parent_id": p}, "hybrid") for p in ("a", "a", "b", "c")]

    class Search:
        parent_documents = parents
        def search(self, query):
            return hits

    class Reranker:
        def rerank(self, query, documents, top_k):
            return [RerankResult(d["text"], d["score"], 1.0, d["metadata"], i + 1)
                    for i, d in enumerate(documents[:top_k])]

    _, contexts = run_query("policy", Search(), Reranker())
    assert contexts == list(parents.values())


def test_failed_evaluation_does_not_replace_existing_report(monkeypatch, tmp_path):
    import src.pipeline as pipeline
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "load_test_set", lambda: [{"question": "q", "ground_truth": "gt"}])
    monkeypatch.setattr(pipeline, "run_query", lambda *args: ("answer", ["context"]))
    monkeypatch.setattr(pipeline, "evaluate_ragas", lambda *args: {"per_question": []})
    monkeypatch.setattr(pipeline, "save_report", lambda *args: pytest.fail("report must be preserved"))
    with pytest.raises(RuntimeError, match="existing report was preserved"):
        pipeline.evaluate_pipeline(None, None)


@pytest.mark.parametrize("failure", ["missing_metric", "quota"])
def test_evaluator_does_not_report_partial_scores_as_valid(monkeypatch, failure):
    from types import SimpleNamespace
    import pandas as pd
    import ragas
    import ragas.embeddings
    import langchain_openai
    import src.m4_eval as m4

    monkeypatch.setattr(m4, "OPENAI_API_KEY", "test-only-no-network")
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: object())
    monkeypatch.setattr(ragas.embeddings, "HuggingfaceEmbeddings", lambda **kwargs: object())

    def evaluate(*args, **kwargs):
        assert kwargs["raise_exceptions"] is True
        if failure == "quota":
            raise RuntimeError("402 INSUFFICIENT_BALANCE")
        frame = pd.DataFrame([{
            "faithfulness": float("nan"), "answer_relevancy": 1.0,
            "context_precision": 1.0, "context_recall": 1.0,
        }])
        return SimpleNamespace(to_pandas=lambda: frame)

    monkeypatch.setattr(ragas, "evaluate", evaluate)
    result = m4.evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    assert result["per_question"] == []
    assert "evaluation_error" in result
