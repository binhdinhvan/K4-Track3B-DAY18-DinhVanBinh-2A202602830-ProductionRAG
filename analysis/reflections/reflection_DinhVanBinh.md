# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Đinh Văn Bình

**MSSV:** 2A202602830

**Khóa:** K4 — Track 3B
**Ngày hoàn thiện:** 04/10/2026

## Phần 1: Mapping bài giảng vào code

| Lecture concept | Module | Hàm cụ thể | Quan sát và phân tích |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()`, `compare_strategies()` | Nhóm các câu liền kề theo cosine similarity với threshold 0,85. Số liệu đo trên corpus được ghi ngay dưới bảng này; không suy ra chất lượng retrieval chỉ từ số chunk. Encoder MiniLM được giữ trong cache của process để tránh nạp lại khi so sánh nhiều tài liệu. |
| Hierarchical chunking | M1 + pipeline | `chunk_hierarchical()`, `build_pipeline()`, `run_query()` | Index child tối đa 256 ký tự để tìm đúng đoạn, rồi đưa parent tối đa 2048 ký tự cho LLM. Parent ID gồm hash của source và nội dung, tránh trùng giữa tài liệu. Các child cùng parent chỉ tạo một context. |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `BM25Search.search()`, `DenseSearch.search()`, `reciprocal_rank_fusion()` | BM25 xử lý từ khóa; Dense dùng bge-m3 và Qdrant. RRF cộng `1/(60 + rank + 1)` để kết hợp thứ hạng, không cộng trực tiếp hai thang điểm khác nhau. Qdrant có fallback in-memory khi dịch vụ local không truy cập được. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | bge-reranker-v2-m3 chấm từng cặp query–child. Đo trước khi sửa cache: nạp model khoảng 6,06 giây, predict 3 docs khoảng 0,19–0,21 giây trên CPU. Chia sẻ model giữa các instance trong cùng process giúp bộ test M3 tránh nạp 5 lần. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Đánh giá 20 câu: Faithfulness 0,8457; Relevancy 0,7582; Precision 0,9917; Recall 0,9333. Relevancy thấp nhất và giảm so với baseline 0,8089, dù recall tăng. Cần xem từng câu; không kết luận nguyên nhân chỉ từ metric thấp nhất hoặc so sánh hai evaluator khác nhau như một ablation. |
| Contextual embeddings | M5 | `contextual_prepend()`, `_enrich_single_call()`, `enrich_chunks()` | Một API call/chunk sinh summary, questions, context và metadata. Pipeline embed child có context prepend; parent nguyên bản được dùng để sinh câu trả lời. Giữ source/parent_id gốc ưu tiên hơn metadata do LLM sinh. Fallback không có API key đã được kiểm tra. |

Kết quả `compare_strategies()` trên nội dung gộp của 26 tài liệu: basic 51 chunks (trung bình 410 ký tự), semantic 208 (99 ký tự), hierarchical 87 children/11 parents (240 ký tự/child), structure-aware 106 (196 ký tự). Pipeline thực tế chunk từng tài liệu riêng và tạo 100 children; không dùng thống kê trên corpus gộp để suy ra số child của pipeline. Với threshold 0,85, semantic tạo nhiều chunk hơn basic trên corpus này; nên thử threshold thấp hơn và đo recall trước khi chọn mặc định.

## Phần 2: Khó khăn và cách giải quyết

### 1. Test chạy chậm do nạp model và gọi API

Lệnh đo là `python -m pytest tests/ -v --durations=10`. Trước khi tối ưu, 37 test pass nhưng mất 294,38 giây; riêng M3 mất 50,67 giây khi chạy độc lập. Kiểm tra cache cho thấy bge-m3 và bge-reranker-v2-m3 đã có weights khoảng 2,27 GB/model. `torch.cuda.is_available()` trả về `False`.

Tách phép đo thành import, load và predict cho thấy suy luận 3 docs chỉ khoảng 0,2 giây. Nguyên nhân lớn của M3 là mỗi test tạo instance mới và nạp lại weights. Đã thêm cache model theo tên; 23 test M1–M3 sau sửa chạy trong 26,85 giây. M4 và M5 vẫn gọi API thật nên thời gian phụ thuộc endpoint; không bỏ việc đánh giá thật để giảm thời gian.

### 2. Parent ID hợp lệ nhưng pipeline vẫn trả context thiếu

Phép kiểm tra trước sửa in đúng kết quả: `Parent length: 400 Returned context length: 80` và `Returns full parent: False`. Đây là lỗi hành vi, không phát sinh exception. Các test cũ kiểm tra parent_id trong một tài liệu nên không phát hiện việc mất parent khi ghép pipeline.

Đã lưu parent trong `search.parent_documents`, khôi phục parent sau reranking và bỏ context trùng. Test hồi quy dùng nhiều child trỏ về cùng parent để kiểm tra LLM vẫn nhận đủ ba parent khác nhau. Child được cắt tại khoảng trắng khi có thể, giảm việc cắt giữa từ.

### 3. Heading trong code block bị hiểu nhầm là section

Ví dụ Markdown chứa `# code comment` bên trong fenced code tạo hai chunk, mỗi chunk chỉ có một dấu mở/đóng fence. Sửa parser để theo dõi fence backtick hoặc tilde; heading bên trong fence không được mở section. Test hồi quy kiểm tra cả hai loại fence.

### 4. Cảnh báo dependency và kiểm tra nộp bài

Cảnh báo thực tế từ RAGAS là:

```text
DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
```

Đây là warning trong dependency, không phải test failure. Cần kiểm tra tương thích RAGAS/sentence-transformers trước khi nâng version; chưa sửa mã của dependency trong lab. Script `check_lab.py` trước đây timeout sau 120 giây và không tính thiếu reflection/test failure vào trạng thái cuối. Đã tăng timeout lên 900 giây, đọc kết quả pytest qua JUnit XML và trả exit code lỗi khi còn thiếu yêu cầu.

Kiến thức cần bổ sung: cách thiết kế kiểm thử ở ranh giới giữa module, đánh giá câu hỏi đa bước, phân biệt lỗi retrieval với lỗi generation hoặc evaluator. Hướng học tiếp là chạy ablation trên cùng test set, lưu context từng câu và kiểm tra thủ công các câu bị RAGAS đánh giá thấp.

### 5. Timeout khi chạy pipeline thực tế

Log của lần chạy sau sửa ghi `Enrichment API failed: Request timed out.` và `LLM generation failed: Request timed out.`. Enrichment dùng fallback cục bộ; generation dùng context đầu tiên khi API lỗi. Đây là cách giữ pipeline hoạt động, nhưng câu trả lời fallback có thể dài và không trả lời đúng trọng tâm. Vì vậy phải kiểm tra answer/context trong báo cáo và không coi exit code 0 là bằng chứng mọi API call đều thành công. Hướng cải tiến tiếp theo là timeout hữu hạn, retry có backoff, lưu kết quả từng bước và gắn trạng thái fallback theo từng câu.

RAGAS sau đó gặp lỗi chính xác: `APIStatusError(Error code: 402 - INSUFFICIENT_BALANCE)` với thông báo `Insufficient balance to complete the request.`. Một số jobs hoàn thành, các jobs khác thất bại. Không dùng trung bình có metric thiếu đổi thành 0 để đánh giá pipeline. Đã giữ snapshot hợp lệ trước sửa, lưu riêng lần chạy thiếu và thêm `raise_exceptions=True` cùng kiểm tra NaN. Sau đổi cấu hình API, request nhỏ trả `OK` và `src/pipeline.py --evaluate-only` hoàn tất đủ 20 câu × 4 metric, không gọi lại enrichment. Một answer fallback vẫn được giữ nguyên và được đánh giá như dữ liệu thực tế, không coi retry là sinh lại tất cả câu trả lời.

## Phần 3: Action plan cho project cá nhân

### Project: RAG hỏi đáp chính sách nội bộ

Project phát triển từ corpus của bài lab: nghỉ phép, lương, bảo mật, mua sắm và đào tạo. Đây là hướng áp dụng được chọn cho action plan, chưa phải hệ thống đã triển khai cho người dùng thực tế.

### Hiện trạng

Pipeline dùng hierarchical chunking → combined enrichment → BM25 + Dense + RRF → CrossEncoder → khôi phục parent → LLM → RAGAS. Hai PDF scan không có text layer được bỏ qua có cảnh báo; chưa có OCR. Corpus có chính sách nhiều phiên bản, nên rủi ro còn lại là chọn bản cũ, thiếu tài liệu cho câu đa bước hoặc hiểu sai cách tính tiền.

### Kế hoạch áp dụng

1. **Chunking:** dùng hierarchical cho tài liệu dài, structure-aware cho Markdown có bảng/section. Bổ sung OCR trước ingestion cho PDF scan. Kiểm tra source và parent_id xuyên suốt pipeline.
2. **Search:** duy trì hybrid + RRF; bổ sung truy vấn con cho câu hỏi kết hợp lương và nghỉ phép. Đo recall@k trước khi tăng top-k.
3. **Reranking:** dùng bge-reranker-v2-m3 đã nạp một lần trong process; đo p50/p95 khi có nhiều request. Chỉ chuyển model nhỏ hơn sau khi so sánh chất lượng và latency trên cùng tập câu hỏi.
4. **Evaluation:** giữ 20 câu của lab làm regression set, bổ sung tối thiểu 10 câu về phiên bản/đa bước/tính toán. Lưu answer, contexts, ground truth và bốn metric từng câu; thêm kiểm tra số tiền/ngày phép bằng giá trị kỳ vọng.
5. **Enrichment:** ưu tiên contextual prepend combined mode, giữ metadata gốc. Thử index thêm hypothesis questions trong một ablation riêng; đánh giá chi phí và chất lượng trước khi bật mặc định.

### Timeline

| Thời gian | Công việc | Tiêu chí hoàn thành |
|---|---|---|
| Tuần 1 — 05–11/10/2026 | OCR, metadata phiên bản/ngày hiệu lực, mở rộng test set | Đọc được PDF scan; thêm ít nhất 10 câu và test chọn chính sách hiện hành |
| Tuần 2 — 12–18/10/2026 | Query decomposition và ablation chunking/search | So sánh trên cùng test set; không làm giảm tính đúng của câu lookup/negation |
| Tuần 3 — 19–25/10/2026 | Benchmark latency, kiểm tra thủ công bottom-5 | Có bảng p50/p95, chi phí/query và ít nhất 3 metric RAGAS ≥ 0,70; ghi rõ trường hợp chưa đạt |

Báo cáo mới đã hoàn tất nằm ở `reports/ragas_report.json`; thời gian thực đo và phân tích được ghi trong `analysis/failure_analysis.md` và `reports/latency_report.json`. `reports/evaluation_inputs.json` cho phép đánh giá lại mà không gọi lại enrichment/generation; `reports/chunking_comparison.json` lưu thống kê chunking. Snapshot trước sửa và lần lỗi quota chỉ được giữ local.
