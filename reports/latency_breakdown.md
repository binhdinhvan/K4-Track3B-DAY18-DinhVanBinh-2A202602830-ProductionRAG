# Latency breakdown — Production RAG

**Ngày đo:** 04/10/2026. **Dữ liệu:** 26 tài liệu đọc được, 100 child chunks, 20 queries. Model chạy CPU; số liệu là một lần chạy, không phải SLA.

## Build và evaluation

| Bước | Thời gian |
|---|---:|
| Load và hierarchical chunking | 0,093 s |
| Combined enrichment, tối đa 4 workers | 414,316 s |
| BM25 + Dense indexing, gồm nạp bge-m3 | 49,759 s |
| Nạp CrossEncoder thực tế | 8,849 s |
| RAGAS 80 jobs | 73,916 s — **không hoàn tất hợp lệ vì hết số dư API** |
| Retry RAGAS bằng API mới trên answers đã lưu | 78,611 s — **hoàn tất 80/80 metric values** |
| RAGAS sau cải tiến generation trên cùng contexts | 57,960 s — **hoàn tất 80/80 metric values** |

## Query latency

Đơn vị: ms. Median là p50; p95 lấy nearest rank trên 20 quan sát.

| Bước | Mean | p50 | p95 |
|---|---:|---:|---:|
| Hybrid retrieval | 183,63 | 182,56 | 215,82 |
| Rerank candidates | 5.369,40 | 5.306,23 | 5.922,47 |
| LLM generation / fallback trước cải tiến | 14.985,97 | 13.254,12 | 23.066,00 |
| Generation mới: draft + verification + arithmetic guard | 4.204,33 | 4.119,50 | 5.425,46 |

Tổng thời gian script quan sát được: **957,7 s**. Generation có một lần timeout và trả context đầu tiên; enrichment có ba thông báo timeout và dùng fallback. Vì vậy các thời gian API bao gồm cả xử lý lỗi, không chỉ latency của response thành công. Rerank ở đây chấm tối đa 20 candidates, khác phép đo 0,2 s với 3 docs của test M3.

## Kết luận từ phép đo

Enrichment chiếm phần lớn thời gian build; nên cache kết quả theo hash nội dung khi phát triển tiếp. Với query, generation là bước chậm nhất, sau đó là CrossEncoder trên CPU. Cần đo chất lượng trước khi giảm candidate count hoặc đổi model.

`latency_report.json` giữ số liệu build/retrieval/rerank gốc, cập nhật generation của cả 20 câu và `ragas_ms` từ lần đánh giá sau cải tiến prompt. Generation mới dùng hai API calls/câu trên contexts đã lưu, có một câu dùng nguồn gốc vì guard phát hiện phép tính sai. Không cộng các phép đo này thành một latency end-to-end mới. Generation ngắn hơn nhưng endpoint và tải API có thể thay đổi; không quy toàn bộ giảm latency cho prompt. Lỗi cũ là `APIStatusError 402 INSUFFICIENT_BALANCE`; metric của lần lỗi không được dùng làm điểm chất lượng. `ragas_report.json` hiện chứa báo cáo mới đầy đủ. Các snapshot lỗi trong `history/` chỉ giữ local, không đưa lên Git.
