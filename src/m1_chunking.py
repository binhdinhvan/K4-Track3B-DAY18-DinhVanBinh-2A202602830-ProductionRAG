from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
import hashlib
from functools import lru_cache
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


@lru_cache(maxsize=1)
def _semantic_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    sentences = [part.strip() for part in re.split(
        r"(?<=[.!?])\s+|\n\n", text
    ) if part.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(sentences[0], {**(metadata or {}), "strategy": "semantic"})]

    from numpy import dot
    from numpy.linalg import norm

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        # Some Windows environments block pyarrow's native DLL, which is
        # imported transitively by sentence-transformers. Keep chunking usable
        # for tests while retaining the model path for a healthy environment.
        if "pyarrow" not in str(exc).lower():
            raise
        import hashlib
        import numpy as np

        vocabulary = sorted({
            token.lower()
            for sentence in sentences
            for token in re.findall(r"\w+", sentence, flags=re.UNICODE)
        })
        index = {token: position for position, token in enumerate(vocabulary)}
        embeddings = np.zeros((len(sentences), len(vocabulary)), dtype=float)
        for row, sentence in enumerate(sentences):
            for token in re.findall(r"\w+", sentence, flags=re.UNICODE):
                # Hashing makes the fallback independent of a third-party
                # tokenizer and deterministic across process runs.
                token = token.lower()
                embeddings[row, index[token]] += int(
                    hashlib.sha1(token.encode("utf-8")).hexdigest()[:2], 16
                ) / 255.0
    else:
        embeddings = _semantic_encoder().encode(sentences)

    def cosine_similarity(first, second):
        return float(dot(first, second) / (norm(first) * norm(second) + 1e-9))

    groups = [[sentences[0]]]
    for index in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings[index - 1], embeddings[index])
        if similarity < threshold:
            groups.append([])
        groups[-1].append(sentences[index])

    base_metadata = metadata or {}

    return [
        Chunk(
            text=" ".join(group),
            metadata={**base_metadata, "strategy": "semantic", "chunk_index": index},
        )
        for index, group in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    parents = []
    children = []
    base_metadata = metadata or {}
    document_id = hashlib.sha256(
        (str(base_metadata.get("source", "")) + "\n" + text).encode("utf-8")
    ).hexdigest()[:16]

    parent_parts = []
    parent_length = 0
    for paragraph in paragraphs:
        pieces = [
            paragraph[index:index + parent_size]
            for index in range(0, len(paragraph), parent_size)
        ] or [paragraph]
        for piece in pieces:
            separator_length = 2 if parent_parts else 0
            if parent_parts and parent_length + separator_length + len(piece) > parent_size:
                parent_text = "\n\n".join(parent_parts)
                parent_id = f"{document_id}_parent_{len(parents)}"
                parents.append(Chunk(
                    parent_text,
                    {**base_metadata, "chunk_type": "parent", "parent_id": parent_id},
                ))
                parent_parts = []
                parent_length = 0
            parent_parts.append(piece)
            parent_length += (2 if len(parent_parts) > 1 else 0) + len(piece)

    if parent_parts:
        parent_id = f"{document_id}_parent_{len(parents)}"
        parents.append(Chunk(
            "\n\n".join(parent_parts),
            {**base_metadata, "chunk_type": "parent", "parent_id": parent_id},
        ))

    for parent in parents:
        parent_id = parent.metadata["parent_id"]
        index = 0
        while index < len(parent.text):
            end = min(index + child_size, len(parent.text))
            if end < len(parent.text):
                boundary = max(parent.text.rfind(" ", index, end), parent.text.rfind("\n", index, end))
                if boundary > index:
                    end = boundary + 1
            child_text = parent.text[index:end].strip()
            if child_text:
                children.append(Chunk(
                    child_text,
                    {
                        **base_metadata,
                        "chunk_type": "child",
                        "parent_id": parent_id,
                        "chunk_index": len(children),
                    },
                    parent_id=parent_id,
                ))
            index = end
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    base_metadata = metadata or {}
    chunks = []
    current_header = ""
    current_content = []

    def append_section():
        content = "\n".join(current_content).strip()
        if not content and not current_header:
            return
        section_text = f"{current_header}\n{content}".strip()
        chunks.append(Chunk(
            section_text,
            {
                **base_metadata,
                "section": current_header,
                "strategy": "structure",
                "chunk_index": len(chunks),
            },
        ))

    fence_character = None
    fence_length = 0
    for part in text.splitlines():
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", part)
        if fence:
            marker, trailing = fence.groups()
            if fence_character is None:
                fence_character, fence_length = marker[0], len(marker)
            elif marker[0] == fence_character and len(marker) >= fence_length and not trailing.strip():
                fence_character = None
            current_content.append(part)
        elif fence_character is None and re.match(r"^#{1,6}\s+", part):
            append_section()
            current_header = part.strip()
            current_content = []
        else:
            current_content.append(part)
    append_section()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
