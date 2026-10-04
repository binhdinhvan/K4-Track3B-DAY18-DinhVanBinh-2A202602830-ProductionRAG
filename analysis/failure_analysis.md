# Failure Analysis — Lab 18: Production RAG

**Học viên:** Đinh Văn Bình — **MSSV:** 2A202602830

**Khóa:** K4 — Track 3B — **Ngày:** 04/10/2026

## Nguồn số liệu và so sánh

Báo cáo hiện tại `reports/ragas_report.json` là lần chạy lại toàn bộ pipeline: 26 tài liệu, 100 chunks, 20 câu × 4 metric đầy đủ. Lưu answer, context, ground truth và metadata evaluator `gpt-4o-mini`. Không truyền ground truth cho retrieval/generation; không sửa ground truth để tăng điểm. Baseline dùng cùng 20 câu nhưng được chấm trước khi đổi cấu hình API, nên so sánh baseline không phải ablation kiểm soát.

| Metric | Naive baseline | Production trước sửa hai câu | Production hiện tại |
|---|---:|---:|---:|
| Faithfulness | 0,8058 | 0,8708 | 0,8708 |
| Answer Relevancy | 0,8089 | 0,7928 | 0,7957 |
| Context Precision | 0,9500 | 1,0000 | 0,9583 |
| Context Recall | 0,8000 | 0,9333 | 0,9583 |

Cả bốn metric ≥ 0,75 và Faithfulness ≥ 0,85, đủ ngưỡng bonus tương ứng. Precision giảm, Recall và Relevancy tăng; không nói tất cả metric tốt hơn. Enrichment được sinh lại và evaluator có dao động, nên không quy toàn bộ thay đổi điểm cho riêng query decomposition. `reports/pipeline_comparison.json` ghi so sánh và hai câu mục tiêu; `answer_prompt_comparison.json` giữ thí nghiệm generation trước đó. Lần lỗi API 402 và thử nghiệm không được chọn chỉ giữ local, không dùng làm điểm chất lượng.

## Diagnostic Tree và cách chọn bottom-5

Chọn trung bình bốn metric thấp nhất, đúng thứ tự `average_score` trong báo cáo. Nhãn tự động chỉ gợi ý, không chứng minh nguyên nhân.

```text
Output sai hoặc thiếu?
├─ Context thiếu → kiểm tra ingestion/OCR, query decomposition, top-k
├─ Retrieve đúng tài liệu nhưng mất đoạn → chunking/parent expansion
├─ Context đủ, đáp án sai → generation, quy tắc tính và số học
├─ Đáp án đúng nhưng metric thấp → đối chiếu claims, ground truth, trace evaluator
└─ API/metric thiếu → dừng evaluation, giữ báo cáo hợp lệ cũ
```

## Bottom-5 failures hiện tại

### #1 — Tạm ứng: số học đúng nhưng thiếu quy tắc phí theo ngày

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày, phí 300.000 VNĐ/tháng; ground truth còn nêu khoảng 50.000 VNĐ theo pro-rata.
- **Got:** Quá hạn 5 ngày (20−15), 15.000.000×2%=300.000 VNĐ/tháng; nêu nguồn chưa quy định cách tính phí riêng cho 5 ngày.
- **Metrics:** Faithfulness 0,2500; Relevancy **0,0000**; Precision 1,0000; Recall 0,6667; trung bình **0,4792**.
- **Error Tree:** Phép tính đúng? → có → nguồn có quy đổi theo ngày? → không → phân biệt giới hạn nguồn với lỗi evaluator.
- **Root cause:** Chính sách tạm ứng hỗ trợ thời hạn 15 ngày và 2%/tháng, không ghi pro-rata; ground truth thêm giả định này. Answer không cung cấp khoản phí ngày cụ thể nên chưa đáp ứng toàn bộ kỳ vọng. Chưa có trace để xác định vì sao faithfulness chỉ 0,25 và relevancy 0; không coi metric thấp là bằng chứng phép tính sai.
- **Suggested fix:** Làm rõ quy tắc phí theo ngày từ chủ sở hữu chính sách; chỉ tính khi có căn cứ. Kiểm tra claim-level trace, đồng thời kiểm thử calculator độc lập. Không tự đổi nguồn/ground truth hoặc khẳng định 50.000 là phí được quy định.

### #2 — Hoàn trả đào tạo: suy luận đúng, context còn nhiễu

- **Question:** Khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành, phải hoàn trả bao nhiêu?
- **Expected / Got:** Hoàn trả 100%, tức 25.000.000 VNĐ, do nghỉ trước cam kết một năm.
- **Metrics:** Faithfulness **0,5000**; Relevancy 0,7561; Precision 0,8333; Recall 1,0000; trung bình **0,7724**.
- **Error Tree:** Context có cam kết/hoàn trả? → có → số tiền đúng? → có → kiểm tra claim suy luận và parent không liên quan.
- **Root cause:** Số tiền và 8 tháng là dữ kiện câu hỏi, quy tắc hoàn trả nằm trong nguồn; cần kiểm chứng evaluator xử lý suy luận này. Context còn một chính sách nghỉ phép, làm giảm precision.
- **Suggested fix:** Nêu căn cứ ngắn “8 < 12 tháng, hoàn trả 100%×25 triệu”; kiểm tra số học riêng, giảm nguồn nhiễu và xem trace trước khi kết luận hallucination.

### #3 — MFA: đáp án có nguồn, recall thấp

- **Question:** Có cần kích hoạt MFA không?
- **Expected / Got:** Có, bắt buộc cho email, VPN và các hệ thống nội bộ theo chính sách hiện hành.
- **Metrics:** Faithfulness 1,0000; Relevancy 0,7632; Precision 1,0000; Recall **0,5000**; trung bình **0,8158**.
- **Error Tree:** Có v2 trong context? → có → đúng MFA? → có → kiểm chứng claims về khác biệt phiên bản trong ground truth.
- **Root cause:** Context gồm bản hiện hành, WFH và bản cũ. Ground truth còn so sánh v1/v2; điểm recall thấp chưa đủ chứng minh nguồn MFA bị thiếu.
- **Suggested fix:** Test ưu tiên phiên bản hiện hành, giữ metadata hiệu lực và kiểm tra trace từng claim. Không chỉ tăng top-k theo nhãn “Missing relevant chunks”.

### #4 — Thiết bị 55 triệu: nguồn phê duyệt xếp sau nguồn nhiễu

- **Question:** Muốn mua thiết bị 55 triệu cần ai phê duyệt?
- **Expected / Got:** Tổng Giám đốc (CEO), vì đơn hàng trên 50 triệu.
- **Metrics:** Faithfulness 1,0000; Relevancy 0,7767; Precision **0,5000**; Recall 1,0000; trung bình **0,8192**.
- **Error Tree:** Đáp án đúng? → có → nguồn đúng đứng đầu? → không → kiểm tra xếp hạng và filtering.
- **Root cause:** Context thứ nhất là tạm ứng; mua sắm đứng thứ hai, hoàn ứng thứ ba. Có đủ căn cứ nhưng thứ hạng nguồn liên quan thấp, phù hợp lỗi precision quan sát được.
- **Suggested fix:** Rerank full parent hoặc dùng tín hiệu tiêu đề/chủ đề, rồi đánh giá trên toàn bộ tập để tránh làm mất recall. Thêm test ngưỡng 50 triệu.

### #5 — Nghỉ không lương: số ngày tình huống cần đối chiếu bảng

- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected / Got:** CEO theo khoảng 16–30 ngày; ground truth thêm nghĩa vụ tự đóng bảo hiểm khi trên 14 ngày.
- **Metrics:** Faithfulness **0,5000**; Relevancy 0,8074; Precision 1,0000; Recall 1,0000; trung bình **0,8269**.
- **Error Tree:** Nguồn có 16–30 ngày/CEO? → có → 20 nằm trong khoảng? → có → kiểm chứng claim-level inference.
- **Root cause:** Đáp án đúng ý trực tiếp nhưng không ghi căn cứ khoảng ngày hoặc lưu ý bảo hiểm. Chưa đủ bằng chứng rằng omission này gây faithfulness thấp; cần trace.
- **Suggested fix:** Nêu ngắn “20 ngày thuộc khoảng 16–30 ngày, CEO phê duyệt”; thêm lưu ý liên quan khi cần hướng dẫn đầy đủ. Test biên khoảng ngày và không gộp thiếu chi tiết với sai sự thật.

## Hai câu mục tiêu và thay đổi đã áp dụng

Câu Senior từng chỉ có các nguồn nghỉ phép. Đã tách hai ý, bỏ dữ kiện thâm niên khỏi truy vấn lương và giữ một parent cho mỗi ý. Context mới có bảng lương và phép năm; đáp án đủ **18 ngày phép, 20–35 triệu VNĐ/tháng**. Metric riêng: Faithfulness 1,0000; Relevancy 0,8813; Precision 1,0000; Recall 1,0000. Senior không còn nằm trong bottom-5.

Câu tạm ứng dùng calculator đọc thời hạn/tỷ lệ từ nguồn, thay vì LLM tự tính hoặc fallback nguyên tài liệu. Không dùng cho tình huống công tác, trả một phần hay chính sách có quy đổi theo ngày. Test thay số tiền/thời hạn/tỷ lệ để kiểm tra không gắn sẵn đáp án lab. Guard AST/Decimal vẫn kiểm tra phương trình parse được cho các câu khác; không đảm bảo phát hiện mọi lỗi suy luận.

Các sửa trước vẫn giữ: parent ID theo nguồn/nội dung, parent expansion, metadata nguồn ưu tiên, cache model, parser fenced code và dừng evaluation thiếu/NaN. 58/58 tests offline pass, kèm RAGAS thật đủ 80/80 giá trị. Combined enrichment một call/chunk; generation thường hai calls/câu, riêng calculator tạm ứng không gọi LLM. Latency đo lại từ toàn bộ pipeline tại `reports/latency_report.json` và `latency_breakdown.md`.

Quy tắc decomposition hiện hỗ trợ hai ý phối hợp và chưa bao quát mọi câu đa bước. Hai PDF scan vẫn cần OCR. Hướng tiếp theo là cải thiện thứ hạng nguồn, decomposition tổng quát, quy tắc tính có cấu trúc và trace evaluator. Báo cáo đạt ngưỡng bonus không đồng nghĩa mọi câu đều hoàn hảo hay mọi lần đánh giá đều cho cùng điểm.
