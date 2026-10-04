# Failure Analysis — Lab 18: Production RAG

**Học viên:** Đinh Văn Bình — **MSSV:** 2A202602830

**Khóa:** K4 — Track 3B — **Ngày phân tích:** 04/10/2026

## Nguồn số liệu

`reports/ragas_report.json` chứa kết quả đầy đủ 20 câu × 4 metric, answers, contexts và ground truth từng câu. Báo cáo hiện tại sinh lại **toàn bộ 20 answers** trên contexts đã lưu từ pipeline có parent expansion. Generation dùng câu hỏi và nguồn, không được cung cấp ground truth; ground truth chỉ dùng cho evaluation. Prompt yêu cầu trả lời ngắn, thêm một lượt kiểm chứng context và guard số học. `reports/answer_prompt_comparison.json` lưu so sánh với generation trước sửa, cùng evaluator model `gpt-4o-mini`.

Baseline tại `reports/naive_baseline_report.json` dùng cùng 20 câu nhưng được chấm trước lần đổi cấu hình API. Vì evaluator có thể khác, so sánh baseline là kết quả quan sát, không phải ablation kiểm soát để quy mọi thay đổi cho một module. Các lần lỗi `402 INSUFFICIENT_BALANCE` không được dùng làm điểm chất lượng. Snapshot và thử nghiệm không được chọn chỉ giữ local trong `reports/history/`.

## RAGAS scores

| Metric | Naive baseline | Production hiện tại | Δ so với baseline |
|---|---:|---:|---:|
| Faithfulness | 0,8058 | 0,8708 | +0,0650 |
| Answer Relevancy | 0,8089 | 0,7928 | −0,0162 |
| Context Precision | 0,9500 | 1,0000 | +0,0500 |
| Context Recall | 0,8000 | 0,9333 | +0,1333 |

Trước thay đổi generation, production đạt Faithfulness 0,8457 và Relevancy 0,7582. Hiện cả bốn metric ≥ 0,75 và Faithfulness ≥ 0,85, đủ ngưỡng hai bonus tương ứng trong rubric. Relevancy vẫn dưới baseline; không kết luận pipeline tốt hơn ở mọi khía cạnh. Precision thực tế là 0,9999999999241667, được làm tròn trong bảng. Contexts giữ nguyên mà precision thay đổi so với lần chấm trước cho thấy evaluator có dao động; không diễn giải thay đổi này thành cải thiện retrieval.

## Cách chọn bottom-5 và Diagnostic Tree

Chọn năm câu có **trung bình bốn metric thấp nhất**, đúng thứ tự `average_score` trong báo cáo. `worst_metric` và nhãn diagnosis tự động chỉ gợi ý debug, không chứng minh nguyên nhân.

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

### #1 — Senior 9 năm: thiếu bảng lương trong retrieval

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 18 ngày phép; lương Senior P3–P4 là 20–35 triệu VNĐ/tháng.
- **Got:** Đúng 18 ngày nhưng nêu không có thông tin cụ thể về lương trong context.
- **Scores:** Faithfulness 0,7500; Relevancy **0,0000**; Precision 1,0000; Recall 0,5000; trung bình **0,5625**.
- **Error Tree:** Output thiếu lương → context có bảng lương? → không → lỗi retrieval cho câu đa bước.
- **Root cause:** Ba parent được retrieve đều về nghỉ phép: năm 2024, năm 2023 và nghỉ không lương. Parent expansion khôi phục nguồn đầy đủ nhưng không bổ sung nguồn độc lập về lương. Chỉ sửa prompt không giải quyết được thiếu dữ liệu này.
- **Suggested fix:** Tách truy vấn ngày phép và khoảng lương Senior, hợp nhất nguồn trước reranking. Thêm kiểm tra context có cả bảng lương và chính sách phép hiện hành; không tự điền đáp án khi thiếu nguồn.

### #2 — MFA: đáp án đúng nhưng recall thấp

- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có; theo v2.0 hiện hành, bắt buộc cho email, VPN và hệ thống nội bộ; v1.0 không yêu cầu MFA.
- **Got:** Nêu đúng yêu cầu bắt buộc và ba nhóm hệ thống.
- **Scores:** Faithfulness 1,0000; Relevancy 0,6649; Precision 1,0000; Recall **0,5000**; trung bình **0,7912**.
- **Error Tree:** Output đúng yêu cầu? → có → context có v2 và MFA? → có → kiểm chứng claims của evaluator và thông tin phiên bản trong ground truth.
- **Root cause:** Context có chính sách hiện hành hỗ trợ câu trả lời; điểm recall thấp chưa đủ chứng minh thiếu nguồn. Ground truth còn mô tả sự khác biệt với v1.0. Cần trace chấm từng claim để phân biệt thiếu bằng chứng phiên bản với biến động evaluator.
- **Suggested fix:** Kiểm tra nguồn/ngày hiệu lực trong contexts và trace recall; thêm test chọn v2 thay v1. Chỉ thêm thông tin phiên bản ngắn khi cần làm rõ chính sách.

### #3 — Tạm ứng: guard chặn phép tính sai, fallback chưa trả lời số tiền

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày; 2%/tháng trên 15 triệu = 300.000 VNĐ/tháng. Ground truth nêu khoảng 50.000 VNĐ nếu tính pro-rata 5 ngày.
- **Got:** Trả nguyên chính sách tạm ứng: thời hạn 15 ngày và mức phí 2%/tháng; không đưa số tiền phạt cụ thể.
- **Scores:** Faithfulness 1,0000; Relevancy **0,5895**; Precision 1,0000; Recall 0,6667; trung bình **0,8141**.
- **Error Tree:** Output thiếu con số → API lỗi? → không → guard phát hiện phương trình sai → fallback nguồn dài, chưa hoàn thành suy luận.
- **Root cause:** LLM vẫn tính sai dù có lượt kiểm chứng, từng đưa `15.000.000 * 0.067% * 5 = 5.000`. Guard bắt lỗi và dùng nguồn gốc. Nguồn chỉ quy định 2%/tháng, không nói cách quy đổi theo ngày; ground truth bổ sung giả định pro-rata. Faithfulness cao của fallback không đồng nghĩa câu hỏi đã được giải quyết đầy đủ.
- **Suggested fix:** Tính số học bằng code có cấu trúc sau khi trích xuất quy tắc có nguồn: 20−15=5 ngày, 15.000.000×2%=300.000/tháng. Chỉ tính phí theo ngày khi chính sách quy định cách quy đổi; nếu thiếu, nêu rõ giới hạn. Làm fallback ngắn và có trạng thái để dễ debug.

### #4 — Hoàn trả đào tạo: suy luận đúng nhưng faithfulness thấp

- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Nghỉ trước cam kết một năm nên hoàn trả 100%, tức 25 triệu VNĐ.
- **Got:** Đúng 100% và 25.000.000 VNĐ.
- **Scores:** Faithfulness **0,5000**; Relevancy 0,7773; Precision 1,0000; Recall 1,0000; trung bình **0,8193**.
- **Error Tree:** Đáp án số đúng? → có → context có cam kết và hoàn trả? → có → kiểm tra chuỗi suy luận và từng claim chấm faithfulness.
- **Root cause:** Chính sách có quy tắc hoàn trả nhưng số 25 triệu và 8 tháng nằm trong câu hỏi. Evaluator có thể không xác nhận đầy đủ suy luận từ dữ kiện tình huống kết hợp nguồn; chưa có trace để kết luận. Nhãn tự động “LLM hallucinating” không đủ chứng minh số tiền sai.
- **Suggested fix:** Diễn giải ngắn “8 tháng < 12 tháng cam kết, hoàn trả 100% × 25 triệu = 25 triệu”; kiểm tra số học riêng và xem claim-level trace. Giữ nguyên ground truth khi so sánh.

### #5 — Laptop 30 triệu: đúng ý chính nhưng thiếu chi tiết đối chiếu

- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Director phê duyệt do nằm trong 5–50 triệu; CNTT xác nhận cấu hình trước đề xuất; ít nhất ba báo giá vì trên 10 triệu.
- **Got:** Director phê duyệt và CNTT xác nhận cấu hình; không nêu ba báo giá.
- **Scores:** Faithfulness **0,5000**; Relevancy 0,7961; Precision 1,0000; Recall 1,0000; trung bình **0,8240**.
- **Error Tree:** Đúng người và xác nhận CNTT? → có → context có ngưỡng phê duyệt? → có → kiểm chứng cách evaluator đối chiếu số tiền tình huống với bảng ngưỡng.
- **Root cause:** Đáp án đáp ứng hai ý được hỏi trực tiếp, nhưng thiếu điều kiện báo giá trong ground truth. Faithfulness thấp có thể liên quan suy luận 30 triệu thuộc khoảng 5–50 triệu; cần trace để xác nhận, không quy thiếu báo giá là nguyên nhân chắc chắn của faithfulness.
- **Suggested fix:** Nêu ngắn căn cứ ngưỡng tiền cùng người phê duyệt; bổ sung điều kiện báo giá liên quan nếu cần hướng dẫn thực hiện đầy đủ. Thêm test các biên 5/10/50 triệu và xác nhận CNTT.

## Thay đổi đã áp dụng và giới hạn

Đã sửa parent expansion, ID không trùng giữa tài liệu, giữ metadata nguồn, cache model, parser fenced code và xử lý evaluation thiếu/NaN. Generation hiện trả lời ngắn, thêm lượt kiểm chứng chỉ dùng nguồn và kiểm tra các phương trình số tường minh bằng AST/Decimal. Hai thử nghiệm prompt trước có điểm cao nhưng sai phép tính nên không được dùng làm báo cáo nộp; thay đổi cuối cùng được đánh giá lại trên cả 20 câu, không chọn riêng các câu tốt.

Guard chỉ kiểm tra phương trình parse được, không kiểm chứng mọi con số hoặc tính đúng của cách áp dụng chính sách. Hai lượt LLM tăng số API calls của generation lên hai/câu; combined enrichment vẫn một/chunk. Câu tạm ứng còn fallback và câu Senior còn thiếu nguồn lương. Cần query decomposition, tính toán có cấu trúc và trace evaluator để xử lý tiếp; ngưỡng bonus hiện đạt không bảo đảm điểm chấm cuối cùng hoặc mọi lần chạy lại đều cho cùng metric.

Latency trong `reports/latency_report.json` giữ build/retrieval/rerank của lần chạy pipeline trước và cập nhật generation/RAGAS từ phép đo mới trên contexts đã lưu. `reports/latency_breakdown.md` ghi rõ đây không phải một lần đo end-to-end mới.
