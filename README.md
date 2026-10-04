# Lab 18: Production RAG Pipeline

**K4-Track3B · Ngày 18 · Production RAG**  
**Thời gian:** 2h implement + 30 phút reflection

## Bài làm cá nhân

- **Học viên:** Đinh Văn Bình — **MSSV:** 2A202602830.
- **Tên repository theo quy chuẩn nộp:** `K4-Track3B-DAY18-DinhVanBinh-2A202602830-ProductionRAG`.
- **GitHub (Public):** [Repository bài nộp](https://github.com/binhdinhvan/K4-Track3B-DAY18-DinhVanBinh-2A202602830-ProductionRAG).
- [Failure analysis và bảng so sánh](analysis/failure_analysis.md).
- [Reflection: mapping, debugging và action plan](analysis/reflections/reflection_DinhVanBinh.md).
- [Báo cáo RAGAS](reports/ragas_report.json) và [baseline](reports/naive_baseline_report.json).
- Số liệu so sánh chunking nằm trong reflection và [JSON thống kê](reports/chunking_comparison.json); [latency breakdown](reports/latency_breakdown.md) có [số liệu gốc](reports/latency_report.json).
- [Answers/contexts đã lưu](reports/evaluation_inputs.json) hỗ trợ `--evaluate-only`; tests hồi quy bổ sung nằm trong `tests/test_pipeline.py`. Biên bản kiểm tra riêng và snapshot chạy lỗi cũ chỉ lưu local.

Kiểm tra bài làm trong môi trường Windows hiện tại:

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -v
.\venv\Scripts\python.exe src/pipeline.py
.\venv\Scripts\python.exe check_lab.py
```

Nếu endpoint API chưa đủ số dư, kiểm tra code/fallback bằng `python check_lab.py --offline`. Dùng `python src/pipeline.py --evaluate-only` để đánh giá lại answers/contexts đã lưu mà không chạy lại enrichment. Lần đánh giá đầu ngày 04/10/2026 gặp `402 INSUFFICIENT_BALANCE`; sau đổi cấu hình API đã hoàn tất 20 câu × 4 metric. Sau cải tiến generation, báo cáo hiện tại đạt Faithfulness **0,8708**; Answer Relevancy **0,7957**; Context Precision **0,9583**; Context Recall **0,9583**. Đạt ngưỡng Faithfulness ≥ 0,85 và cả bốn metric ≥ 0,75 theo rubric; điểm cuối cùng do giảng viên chấm. Checker sẽ báo chưa sẵn sàng nếu một lần evaluation mới còn thiếu.

Generation trả lời ngắn, dùng thêm một lượt kiểm chứng dựa trên context và kiểm tra số học cho các phương trình số tường minh. Khi phát hiện phép tính sai, pipeline trả nguồn gốc thay vì số tiền sai. Guard chỉ kiểm tra những phương trình parse được, không chứng minh mọi suy luận đều đúng. `python scripts/evaluate_answer_prompt.py` sinh lại toàn bộ 20 answers trên contexts đã lưu rồi đánh giá; không truyền ground truth cho generation và không tự ghi đè báo cáo nộp. [So sánh generation](reports/answer_prompt_comparison.json) ghi rõ điều kiện và giới hạn phép đo.

Pipeline retrieve/rerank child, khôi phục parent gốc và loại context trùng trước khi gọi LLM. Câu hỏi có hai ý phối hợp được tách truy vấn và giữ một nguồn cho mỗi ý: câu Senior hiện trả lời đủ 18 ngày phép và lương 20–35 triệu/tháng. Với tình huống tạm ứng đơn giản, code lấy thời hạn và tỷ lệ phí từ nguồn để tính số ngày quá hạn và phí tháng, không tự quy đổi theo ngày. Câu tạm ứng hiện trả 5 ngày quá hạn, 300.000 VNĐ/tháng và nêu thiếu quy tắc phí theo ngày. [So sánh pipeline mới](reports/pipeline_comparison.json) ghi rõ Precision giảm, Recall tăng và Faithfulness giữ nguyên; `python scripts/evaluate_pipeline_candidate.py` chạy lại toàn bộ pipeline vào thư mục local để thử nghiệm trước khi thay báo cáo nộp.

Báo cáo lưu cả context và metric từng câu để kiểm tra nguyên nhân lỗi. Hai PDF scan cần OCR được bỏ qua có cảnh báo; OCR chưa nằm trong pipeline này. API enrichment lỗi sẽ dùng fallback, còn RAGAS không đánh giá đủ câu sẽ dừng và giữ nguyên báo cáo cũ. Câu tạm ứng dùng calculator không gọi LLM; các câu khác dùng draft và verification. Bảng so sánh generation trước đó là kết quả lịch sử, không phải báo cáo hiện tại.

---

## Tổng quan

Bài tập **cá nhân** — implement toàn bộ 5 modules:

```
M1 Chunking → M5 Enrichment → M2 Hybrid Search → M3 Reranking → LLM Answer → M4 RAGAS Eval
```

Xem **ASSIGNMENT.md** để biết chi tiết từng module và timeline.

## Prerequisites

| Dependency | Bắt buộc? | Dùng cho |
|-----------|-----------|----------|
| Docker (Qdrant) | ✅ Có | M2 Dense Search |
| Python 3.11+ | ✅ Có | Tất cả modules (RAGAS cần 3.11+ cho asyncio) |
| `OPENAI_API_KEY` | ⚠️ M4+M5 | RAGAS eval (M4), Enrichment LLM (M5) |

**Pre-download models** (tránh timeout trong lab):
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
```

## Quick Start

### 1. Clone repository & tạo môi trường ảo

**Linux / macOS / Git Bash:**
```bash
git clone <repo-url>
cd K4-Track3B-Production-RAG
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
git clone <repo-url>
cd K4-Track3B-Production-RAG
python -m venv .venv
.venv\Scripts\Activate.ps1
```
*(Nếu dùng Windows CMD: chạy `.venv\Scripts\activate.bat`)*

### 2. Cài đặt dependencies & Khởi động dịch vụ

**Linux / macOS / Git Bash:**
```bash
docker compose up -d                    # Khởi động Qdrant vector database
pip install -r requirements.txt
cp .env.example .env                    # Tạo file .env và điền OPENAI_API_KEY
python naive_baseline.py                # Khởi tạo baseline
```

**Windows (PowerShell):**
```powershell
docker compose up -d                    # Khởi động Qdrant vector database
pip install -r requirements.txt
Copy-Item .env.example .env             # Tạo file .env và điền OPENAI_API_KEY
python naive_baseline.py                # Khởi tạo baseline
```
*(Nếu dùng Windows CMD: dùng `copy .env.example .env` thay cho `Copy-Item`)*

## Chạy toàn bộ & Kiểm tra

```bash
python main.py                          # Chạy Naive + Production + In bảng so sánh
python check_lab.py                     # Script kiểm tra hợp lệ trước khi nộp (chạy được trên mọi OS)
```

## Cấu trúc repo

```
K4-Track3B-Production-RAG/
├── README.md                   # File này
├── ASSIGNMENT.md               # ★ Đề bài + timeline + reflection
├── RUBRIC.md                   # Hệ thống chấm điểm
│
├── main.py                     # Entry point: chạy toàn bộ pipeline
├── check_lab.py                # Kiểm tra định dạng trước khi nộp
├── naive_baseline.py           # Baseline (chạy trước)
├── config.py                   # Shared config
├── requirements.txt            # Dependencies
├── docker-compose.yml          # Qdrant local
├── .env.example                # API keys template
│
├── data/                       # Corpus tiếng Việt — 25 .md files + 3 PDFs (28 files total)
│   ├── nghi_phep_nam_v2023.md  # Nghỉ phép 12 ngày (v2023, superseded)
│   ├── nghi_phep_nam_v2024.md  # Nghỉ phép 15 ngày (v2024, hiện hành)
│   ├── mat_khau_v1.md          # Password policy 90 ngày (OLD)
│   ├── mat_khau_v2.md          # Password policy 120 ngày + MFA (NEW)
│   ├── ... (28 files total)    # 8 categories: leave, salary, IT, workflow, training, admin, safety, compliance
│   ├── so_tay_an_toan.pdf      # An toàn PCCC + sơ cứu (PDF text)
│   ├── BCTC.pdf                # Báo cáo tài chính (scan, cần OCR)
│   └── Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf # Nghị định BVDL (scan, cần OCR)
├── test_set.json               # 20 Q&A pairs (6 types: lookup, version, negation, multi-hop, numeric, ambiguous)
│
├── src/                        # ★ Scaffold code (có TODO markers)
│   ├── m1_chunking.py          # Module 1: Chunking
│   ├── m2_search.py            # Module 2: Hybrid Search
│   ├── m3_rerank.py            # Module 3: Reranking
│   ├── m4_eval.py              # Module 4: Evaluation
│   ├── m5_enrichment.py        # Module 5: Enrichment Pipeline
│   └── pipeline.py             # Ghép toàn bộ pipeline
│
├── tests/                      # Auto-grading
│   ├── test_m1.py
│   ├── test_m2.py
│   ├── test_m3.py
│   ├── test_m4.py
│   └── test_m5.py
│
├── analysis/                   # ★ Deliverable
│   ├── failure_analysis.md     # Phân tích failures (cá nhân)
│   └── reflections/            # Reflection cá nhân
│       └── reflection_TEMPLATE.md
│
├── reports/                    # ★ Auto-generated (bắt buộc: reports/ragas_report.json)
│   ├── ragas_report.json
│   └── naive_baseline_report.json
│
└── templates/                  # Templates gốc (backup)
    └── failure_analysis.md
```

## Timeline (Thời lượng ước tính)

| Thời lượng | Hoạt động |
|------------|-----------|
| 10 phút | Setup môi trường + chạy `naive_baseline.py` |
| 90 phút | Implement M1 → M2 → M3 → M4 → M5 |
| 20 phút | Chạy pipeline + RAGAS + failure analysis |
| 30 phút | Reflection: lecture mapping + project plan |

## Quy chuẩn đặt tên Repository & Nộp bài

- **Cấu trúc đặt tên repo:**  
  `K4-Track3B-DAY18-<HoVaTen>-<MSSV>-ProductionRAG`  
  *(Ví dụ: `K4-Track3B-DAY18-NguyenVanAn-AI20K001-ProductionRAG`)*
- **Hạn chót nộp bài:** **11h59 ngày hôm sau diễn ra bài lab (GMT+7)** trên cổng VLearn LMS / Codelab.
- **Chi tiết yêu cầu:** Xem tại [ASSIGNMENT.md](ASSIGNMENT.md) và [RUBRIC.md](RUBRIC.md).
