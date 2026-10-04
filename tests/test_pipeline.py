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


def test_grounded_generation_uses_only_question_and_context():
    from types import SimpleNamespace
    from src.pipeline import generate_grounded_answer
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  15 ngày.  "))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    answer = generate_grounded_answer("Ngày phép?", ["Chính sách: 15 ngày."], client)
    assert answer == "15 ngày."
    assert len(calls) == 2
    assert calls[0]["temperature"] == 0
    assert calls[0]["messages"][1]["content"] == "Context:\nChính sách: 15 ngày.\n\nCâu hỏi: Ngày phép?"
    assert "Bản nháp:" in calls[1]["messages"][1]["content"]


def test_grounded_generation_without_context_does_not_call_api():
    from src.pipeline import generate_grounded_answer
    assert generate_grounded_answer("question", [], object()) == "Không tìm thấy thông tin."


def test_grounded_generation_rejects_empty_completion():
    from types import SimpleNamespace
    from src.pipeline import generate_grounded_answer
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" "))])
    )))
    with pytest.raises(RuntimeError, match="empty answer"):
        generate_grounded_answer("question", ["context"], client)


def test_grounded_generation_rejects_empty_verification():
    from types import SimpleNamespace
    from src.pipeline import generate_grounded_answer
    outputs = iter(("Draft answer", ""))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(outputs)))])
    )))
    with pytest.raises(RuntimeError, match="empty verified answer"):
        generate_grounded_answer("question", ["context"], client)


@pytest.mark.parametrize("answer,valid", [
    ("15.000.000 VNĐ * 2% = 300.000 VNĐ", True),
    ("15.000.000 VNĐ * 0.067% * 5 ngày = 5.000 VNĐ", False),
    ("15.000.000 * 2% * (20/30) = 1.000.000", False),
    ("20.000.000 * 85% = 17.000.000", True),
    ("10 / 0 = 0", False),
])
def test_explicit_arithmetic_is_checked(answer, valid):
    from src.pipeline import _calculations_match
    assert _calculations_match(answer) is valid


def test_wrong_arithmetic_returns_source_instead_of_wrong_amount():
    from types import SimpleNamespace
    from src.pipeline import generate_grounded_answer
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="2 + 2 = 5"))])
    )))
    assert generate_grounded_answer("question", ["Original source"], client) == "Original source"


def test_advance_fee_uses_source_rate_without_daily_assumption():
    from src.pipeline import calculate_advance_fee
    policy = "# Chính sách tạm ứng\nThanh toán trong vòng **10 ngày**. Quá hạn tính phí **3%/tháng**."
    answer = calculate_advance_fee("Tạm ứng 8 triệu, sau 14 ngày thanh toán", [policy])
    assert "Quá hạn 4 ngày" in answer
    assert "240.000 VNĐ/tháng" in answer
    assert "chưa thể xác định khoản phí riêng" in answer
    assert calculate_advance_fee("Tạm ứng 8 triệu công tác, sau 14 ngày", [policy]) is None
    assert calculate_advance_fee("Tạm ứng 8 triệu, đã trả một phần, sau 14 ngày", [policy]) is None
    assert calculate_advance_fee("Tạm ứng 8 triệu, sau 14 ngày", [policy + " Tính theo ngày."]) is None


def test_multihop_retrieval_keeps_each_facet_source():
    from src.pipeline import retrieve_contexts, decompose_query
    query = "Nhân viên Senior có 6 năm thâm niên được nghỉ bao nhiêu ngày phép và lương trong khoảng nào?"
    facets = decompose_query(query)
    assert len(facets) == 2 and "Senior" in facets[1]
    assert "ngày phép" not in facets[1]
    assert "thâm niên" not in facets[1]
    class Search:
        parent_documents = {"leave": "Leave policy", "old": "Old leave", "salary": "Salary table"}
        def search(self, question):
            ids = ["salary", "leave"] if question == facets[1] else ["leave", "old"]
            return [SearchResult(p, 1.0, {"parent_id": p}, "hybrid") for p in ids]
    class Reranker:
        def rerank(self, query, docs, top_k):
            return [RerankResult(d["text"], d["score"], 1.0, d["metadata"], i+1) for i,d in enumerate(docs)]
    contexts, _ = retrieve_contexts(query, Search(), Reranker())
    assert contexts[:2] == ["Leave policy", "Salary table"]
