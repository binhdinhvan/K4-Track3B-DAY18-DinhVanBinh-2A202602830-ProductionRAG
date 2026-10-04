# Failure Analysis — Lab 18: Production RAG

**Học viên:** Đinh Văn Bình — **MSSV:** 2A202602830

**Khóa:** K4 — Track 3B

**Ngày phân tích:** 04/10/2026

## Nguồn số liệu

Bảng và bottom-5 dưới đây dùng **báo cáo mới đã hoàn tất 20 câu × 4 metric** tại `reports/ragas_report.json`. Đánh giá chạy bằng `src/pipeline.py --evaluate-only` trên answers/contexts đã lưu từ pipeline sau sửa parent retrieval. Báo cáo lưu metric và context từng câu; thông tin evaluator nằm trong `evaluation_metadata`.

Baseline có cùng 20 câu tại `reports/naive_baseline_report.json`. Baseline được đánh giá trước lần đổi cấu hình API; vì evaluator có thể khác, bảng này là so sánh kết quả quan sát được, không phải thí nghiệm kiểm soát để quy toàn bộ thay đổi điểm cho một module.

Lần đánh giá trước đó bị `402 INSUFFICIENT_BALANCE` và không được dùng làm điểm chất lượng. Snapshot trước sửa và lần lỗi được giữ local để debug, không thuộc bài nộp; báo cáo hiện nộp là kết quả mới đầy đủ.

## RAGAS scores

| Metric | Naive baseline | Production mới | Δ |
|---|---:|---:|---:|
| Faithfulness | 0,8058 | 0,8457 | +0,0398 |
| Answer Relevancy | 0,8089 | 0,7582 | −0,0508 |
| Context Precision | 0,9500 | 0,9917 | +0,0417 |
| Context Recall | 0,8000 | 0,9333 | +0,1333 |

Cả bốn metric vượt 0,75; đạt ngưỡng rubric ≥ 3 metrics đạt 0,70. Recall tăng nhưng relevancy giảm so với baseline, nên chưa thể nói pipeline tốt hơn ở mọi khía cạnh. Cần tập trung vào câu trả lời dài, câu đa bước và cách kiểm chứng phép tính của evaluator.

## Cách chọn bottom-5 và Diagnostic Tree

Chọn năm câu có **trung bình bốn metric thấp nhất**, đúng thứ tự `average_score` trong báo cáo. `worst_metric` chỉ là gợi ý debug, không tự chứng minh nguyên nhân.

```text
Output sai hoặc thiếu?
├─ Context thiếu thông tin cần thiết
│  ├─ Chưa ingest tài liệu → dữ liệu/OCR
│  ├─ Có trong corpus nhưng không retrieve → search/decomposition/top-k
│  └─ Mất section dù retrieve đúng document → chunking/parent expansion
├─ Context đủ nhưng output sai/thiếu → generation/prompt/tính toán
├─ Output đúng nhưng metric thấp → kiểm chứng claims/evaluator/ground truth
└─ API/evaluator lỗi → xử lý hạ tầng, không gán metric thiếu thành 0
```

## Bottom-5 failures

### #1 — Ngày phép năm: trả lời quá dài và thêm điều kiện thiếu thông tin

- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Theo v2024, 15 ngày phép năm có lương; v2023 quy định 12 ngày nhưng đã bị thay thế.
- **Got:** Nêu đúng 15 ngày, rồi thêm công thức thâm niên, ví dụ 9 năm/18 ngày và kết luận câu hỏi thiếu trạng thái hợp đồng/thâm niên.
- **Worst metric:** Answer Relevancy = **0,0000**; Faithfulness = 0,6667; trung bình = **0,6667**.
- **Error Tree:** Context đúng? → có v2024 → answer có con số đúng? → có → phần diễn giải thêm có làm lệch trọng tâm? → cần kiểm tra generation/evaluator.
- **Root cause:** Answer dài và thêm đoạn “thông tin còn thiếu” dù có thể trả lời mức cơ bản. Điểm 0 có thể liên quan cách evaluator nhận diện câu trả lời không dứt khoát; chưa có trace chi tiết để xác nhận. Không coi 0 là bằng chứng rằng con số 15 sai.
- **Suggested fix:** Với lookup, trả lời ngắn “15 ngày/năm theo v2024”; chỉ thêm thâm niên khi được hỏi. Chỉ báo thiếu thông tin khi thực sự cần để trả lời ý chính; lưu trace evaluator để kiểm chứng vì sao chấm relevancy 0.

### #2 — Senior 9 năm: thiếu bảng lương trong retrieval

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 18 ngày phép; Senior P3–P4 hưởng 20–35 triệu VNĐ/tháng.
- **Got:** Đúng 18 ngày nhưng báo context không có khoảng lương.
- **Worst metric:** Faithfulness = **0,5000**; Recall = 0,5000; trung bình = **0,6779**.
- **Error Tree:** Output thiếu khoảng lương → context có bảng lương? → không → search cho từng ý.
- **Root cause:** Cả ba parent retrieve là nghỉ phép năm 2024, nghỉ phép năm 2023 và nghỉ không lương. Câu đa bước bị chi phối bởi ý nghỉ phép; parent expansion chưa giải quyết việc thiếu một nguồn độc lập. Nhãn tự động “LLM hallucinating” không mô tả đầy đủ lỗi retrieval đã quan sát.
- **Suggested fix:** Tách query về ngày phép và khoảng lương Senior; hợp nhất nguồn trước reranking. Thêm test context phải chứa cả chính sách nghỉ phép và bảng lương, kèm kiểm tra phiên bản hiện hành.

### #3 — Lương thử việc Junior: đáp án số đúng nhưng faithfulness thấp

- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** 20.000.000 × 85% = 17.000.000 VNĐ/tháng.
- **Got:** Đúng mức 17.000.000 và công thức 20 triệu × 85%.
- **Worst metric:** Faithfulness = **0,3333**; Recall = 1,0000; trung bình = **0,7864**.
- **Error Tree:** Output đúng số kỳ vọng? → có → context có bảng lương/thử việc? → có → kiểm chứng từng claim và cách evaluator xử lý phép tính.
- **Root cause:** Hai parent liên quan được retrieve, nên không đủ cơ sở kết luận thiếu context hay hallucination. Có thể evaluator không công nhận một số diễn giải/phép tính; cần trace chấm từng claim để xác nhận.
- **Suggested fix:** Trích ngắn hai dữ kiện nguồn và một dòng phép tính; bổ sung test số tiền kỳ vọng. Kiểm tra thủ công các claim thay vì sửa câu trả lời đúng chỉ để tăng metric.

### #4 — Tạm ứng 15 triệu: quy tắc pro-rata chưa rõ

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected trong test set:** Quá hạn 5 ngày; 300.000 VNĐ/tháng, khoảng 50.000 VNĐ nếu chia pro-rata theo tháng 30 ngày.
- **Got:** Tính 300.000 VNĐ rồi ghi chú rằng context không quy định chia theo ngày.
- **Worst metric:** Answer Relevancy = **0,6654**; Recall = 0,6667; trung bình = **0,8018**.
- **Error Tree:** Có thời hạn/mức phí? → có → có quy tắc chia theo ngày? → chưa rõ → kiểm chứng phạm vi ground truth và generation.
- **Root cause:** Ground truth giả định pro-rata, nhưng chính sách tạm ứng chỉ ghi 2%/tháng. Answer cũng kết luận khoản phạt 300.000 VNĐ quá dứt khoát rồi mới chú thích giới hạn, chưa phân biệt rõ mức theo tháng với số tiền cho 5 ngày.
- **Suggested fix:** Nêu `20 − 15 = 5 ngày`, `15.000.000 × 2% = 300.000 VNĐ/tháng`. Chỉ đưa `300.000 × 5/30 = 50.000 VNĐ` dưới dạng giả định rõ ràng, hoặc bổ sung quy tắc chính thức vào corpus/ground truth. Không tạo thêm chính sách để khớp evaluator.

### #5 — Hoàn chi đào tạo: kiểm chứng nhãn hallucination

- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Nghỉ trước cam kết 1 năm nên hoàn trả 100%, tức 25.000.000 VNĐ.
- **Got:** Đúng số tiền 25 triệu, điều kiện 1 năm và công thức nhân 100%.
- **Worst metric:** Faithfulness = **0,5714**; Recall = 1,0000; trung bình = **0,8023**.
- **Error Tree:** Output khác ground truth? → không → context có quy tắc cam kết/hoàn trả? → có → kiểm tra claim suy luận và evaluator.
- **Root cause:** Số tiền và thời gian xuất phát một phần từ câu hỏi, còn quy tắc từ context. Evaluator có thể chấm thấp một số bước suy luận, nhưng chưa có trace để xác nhận. Không thể khẳng định hallucination chỉ từ nhãn tự động.
- **Suggested fix:** Viết ngắn quy tắc và phép tính, giữ rõ nguồn của dữ kiện; dùng kiểm tra số tiền kỳ vọng bên cạnh RAGAS và kiểm chứng thủ công các claim bị chấm thấp.

## Case study: Laptop 30 triệu

Trước sửa, answer thiếu người phê duyệt dù nêu được xác nhận CNTT. Debug cho thấy `build_pipeline()` bỏ parents và `run_query()` trả child; ví dụ parent 400 ký tự chỉ trả context 80 ký tự.

Sau sửa, context chứa toàn bộ parent “Quy trình mua sắm” và answer nêu đúng Director cùng xác nhận cấu hình CNTT. Test hồi quy kiểm tra khôi phục parent và lấy ba parent khác nhau dù hai child cùng parent đứng đầu. Đây là cải thiện hành vi quan sát được; không quy toàn bộ thay đổi metric cho parent expansion vì evaluator/config API và prompt cũng có thể khác.

## Hạ tầng, thay đổi và giới hạn

- API mới đã qua request kiểm tra và hoàn tất 80 jobs RAGAS. Metric thiếu/exception sẽ dừng evaluation và giữ báo cáo hợp lệ.
- Một answer trong dữ liệu đã lưu dùng fallback do generation timeout: câu Mentor/Buddy. Retry chỉ đánh giá lại answers/contexts, không sinh lại câu này; báo cáo mới phản ánh đúng dữ liệu đó.
- Đã sửa parent expansion, ID trùng, cắt child tại khoảng trắng, fenced code, cache model và bảo toàn metadata gốc.
- Chưa OCR hai PDF scan, chưa lọc metadata theo ngày hiệu lực, chưa xử lý query decomposition đa bước.

## Nếu có thêm một giờ

1. Thử prompt ngắn cho lookup, sinh lại câu Mentor/Buddy và lưu trace evaluator ở các câu tính toán.
2. Thêm query decomposition cho câu Senior, kiểm tra đủ hai nguồn context.
3. Thống nhất quy tắc pro-rata và phạm vi ground truth với người sở hữu dữ liệu; chạy ablation cùng evaluator để so sánh công bằng.

## Latency breakdown

Đo trên CPU với 26 tài liệu, 100 child chunks và 20 queries. Số liệu build/query thuộc lần chạy sinh answers đã lưu; evaluation cuối là lần retry dùng API mới. Không cộng hai lần chạy thành một latency end-to-end mới.

| Bước build/evaluation | Thời gian |
|---|---:|
| Load và chunking | 0,093 s |
| Combined enrichment, tối đa 4 workers | 414,316 s |
| BM25 + Dense indexing, gồm nạp bge-m3 | 49,759 s |
| Nạp CrossEncoder | 8,849 s |
| Retry RAGAS thành công, 80 metric values | 78,611 s |

| Bước query | Mean (ms) | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|
| Hybrid retrieval | 183,63 | 182,56 | 215,82 |
| Rerank candidates | 5.369,40 | 5.306,23 | 5.922,47 |
| Generation/fallback | 14.985,97 | 13.254,12 | 23.066,00 |

p95 dùng nearest rank trên 20 quan sát. Enrichment có ba thông báo timeout, generation có một lần timeout; thời gian gồm cả xử lý lỗi. Đây là một phép đo, không phải SLA. Enrichment là bước build chậm nhất; generation là bước query chậm nhất. Kế hoạch project nằm trong reflection cá nhân.
