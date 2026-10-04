from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    parent_documents = {}
    timings = {}
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        parent_documents.update({parent.metadata["parent_id"]: parent.text for parent in parents})
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)
    timings["chunking_ms"] = (time.time() - t0) * 1000

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks, max_workers=4)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)
    timings["enrichment_ms"] = (time.time() - t0) * 1000

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.parent_documents = parent_documents
    search.build_timings = timings
    search.query_timings = []
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)
    timings["indexing_ms"] = (time.time() - t0) * 1000

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()
    timings["reranker_load_ms"] = (time.time() - t0) * 1000
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    start = time.perf_counter()
    results = search.search(query)
    retrieval_ms = (time.perf_counter() - start) * 1000
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    start = time.perf_counter()
    reranked = reranker.rerank(query, docs, top_k=len(docs))
    rerank_ms = (time.perf_counter() - start) * 1000
    contexts = []
    seen = set()
    parent_documents = getattr(search, "parent_documents", {})
    for result in reranked or results:
        context = parent_documents.get(result.metadata.get("parent_id"), result.text)
        if context not in seen:
            seen.add(context)
            contexts.append(context)
        if len(contexts) >= RERANK_TOP_K:
            break

    from config import OPENAI_API_KEY, OPENAI_MODEL, create_openai_client
    start = time.perf_counter()
    if OPENAI_API_KEY and contexts:
        try:
            client = create_openai_client()
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(model=OPENAI_MODEL, temperature=0, messages=[
                {"role": "system", "content": "Trả lời CHỈ dựa trên context. Ưu tiên chính sách hiện hành khi có nhiều phiên bản. Trả lời đủ từng ý của câu hỏi; được tính toán từ số liệu có trong context và phải ghi công thức. Nếu thiếu thông tin, nêu rõ phần còn thiếu."},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    if hasattr(search, "query_timings"):
        search.query_timings.append({
            "question": query, "retrieval_ms": retrieval_ms, "rerank_ms": rerank_ms,
            "generation_ms": (time.perf_counter() - start) * 1000,
        })
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    os.makedirs("reports", exist_ok=True)
    with open("reports/evaluation_inputs.json", "w", encoding="utf-8") as f:
        json.dump({"questions": questions, "answers": answers, "contexts": all_contexts,
                   "ground_truths": ground_truths}, f, ensure_ascii=False, indent=2)
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    if len(results.get("per_question", [])) != len(test_set):
        raise RuntimeError("RAGAS did not evaluate all questions; existing report was preserved: "
                           + results.get("evaluation_error", "incomplete evaluation"))
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []), bottom_n=5)
    save_report(results, failures)
    latency = {
        "build_ms": getattr(search, "build_timings", {}),
        "ragas_ms": (time.time() - t0) * 1000,
        "queries": getattr(search, "query_timings", []),
    }
    with open("reports/latency_report.json", "w", encoding="utf-8") as f:
        json.dump(latency, f, ensure_ascii=False, indent=2)
    return results


def evaluate_saved_answers():
    """Retry RAGAS using saved answers without repeating enrichment/generation."""
    with open("reports/evaluation_inputs.json", encoding="utf-8") as f:
        inputs = json.load(f)
    start = time.perf_counter()
    results = evaluate_ragas(**inputs)
    if len(results.get("per_question", [])) != len(inputs["questions"]):
        raise RuntimeError("RAGAS retry failed; existing report was preserved: "
                           + results.get("evaluation_error", "incomplete evaluation"))
    save_report(results, failure_analysis(results["per_question"], bottom_n=5))
    path = "reports/latency_report.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            latency = json.load(f)
        latency["ragas_ms"] = (time.perf_counter() - start) * 1000
        latency["evaluation_status"] = "complete"
        latency.pop("evaluation_error", None)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(latency, f, ensure_ascii=False, indent=2)
    return results


if __name__ == "__main__":
    start = time.time()
    if "--evaluate-only" in sys.argv:
        evaluate_saved_answers()
    else:
        search, reranker = build_pipeline()
        evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
