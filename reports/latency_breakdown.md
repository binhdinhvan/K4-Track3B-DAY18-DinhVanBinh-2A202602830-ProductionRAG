# Latency breakdown — Production RAG

**Ngày đo:** 04/10/2026. Chạy lại pipeline thực tế sau sửa query decomposition và calculator tạm ứng: 26 tài liệu, 100 child chunks, 20 câu, model chạy CPU. Đây là một lần đo, không phải SLA. Hai PDF scan chưa có OCR được bỏ qua có cảnh báo.

## Build và evaluation

| Bước | Thời gian |
|---|---:|
| Load + hierarchical chunking | 0,150 s |
| Combined enrichment, tối đa 4 workers | 95,117 s |
| BM25 + Dense indexing, gồm nạp bge-m3 | 41,426 s |
| Nạp CrossEncoder | 6,039 s |
| RAGAS, đủ 80/80 metric values | 39,488 s |

## Query latency

Đơn vị ms; p50 là median, p95 là nearest rank trên 20 quan sát.

| Bước | Mean | p50 | p95 |
|---|---:|---:|---:|
| Hybrid retrieval | 173,49 | 133,21 | 438,51 |
| Rerank candidates | 5.712,62 | 4.776,11 | 13.131,01 |
| Generation / source-based calculator | 3.328,58 | 3.193,20 | 4.620,96 |

Với câu phối hợp hai ý, pipeline tìm kiếm/rerank câu đầy đủ và hai truy vấn con, rồi giữ một parent cho mỗi ý. Vì vậy p95 rerank tăng so với bản trước 5.922,47 ms. Câu tạm ứng dùng code tính từ quy tắc nguồn, không gọi LLM; các câu khác dùng hai API calls cho draft và verification. API generation và enrichment không ghi nhận timeout trong lần chạy này.

`reports/latency_report.json` chứa build và queries từ cùng lần chạy mới, thay cho bảng cũ ghép retrieval/rerank lịch sử với generation mới. RAGAS được chạy sau generation; kiểm tra offline bắt đầu trong giai đoạn evaluation nên có thể ảnh hưởng nhẹ thời gian này. Không coi tổng các bước là thời gian process chính xác vì không bao gồm mọi overhead. Endpoint/tải API và enrichment có thể dao động; không quy toàn bộ thay đổi latency cho code.

Reranking CPU là phần lớn thời gian query. Hướng tiếp theo: batch/cache, giới hạn candidates sau khi đo recall và kiểm tra lựa chọn nguồn cho từng ý. M5 vẫn combined mode một call/chunk; hai calls/câu thuộc generation, không thuộc enrichment. Snapshot và thử nghiệm trước sửa chỉ giữ local trong `reports/history/`.
