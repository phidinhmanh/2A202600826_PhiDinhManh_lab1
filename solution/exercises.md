# Ngày 1 — Bài Tập & Phản Ánh
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Triển khai tất cả TODO trong `template.py`. Chạy `pytest tests/` để kiểm tra tiến độ.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python template.py
```
Bạn sẽ thấy output so sánh phản hồi của GPT-4o và GPT-4o-mini.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature tăng dần từ 0.0 lên 1.5, các phản hồi dịch chuyển từ tính trạng cực kỳ nhất quán, trực diện và chính xác về mặt thông tin (ở mức 0.0) sang ngôn từ phong phú, sinh động hơn (ở mức 0.5 - 1.0). Tuy nhiên, khi tăng lên mức quá cao như 1.5, văn bản bắt đầu xuất hiện lỗi lặp từ vô nghĩa, cấu trúc câu hỗn loạn và có thể đưa ra thông tin sai lệch (ảo tưởng).

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Cho chatbot hỗ trợ khách hàng, tôi sẽ thiết lập temperature ở mức thấp, cụ thể từ **0.0 đến 0.2**. Điều này đảm bảo tính nhất quán tuyệt đối về thông tin (như chính sách, hướng dẫn kỹ thuật), tăng tính tin cậy và hạn chế tối đa hành vi sinh câu chữ ngẫu hứng hoặc sai lệch từ hệ thống.

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này:**
> Workload hàng ngày là 30.000 lượt gọi, tổng cộng 10.500.000 tokens đầu ra. Chi phí chạy GPT-4o là $105.00/ngày ($0.010/1K tokens), trong khi GPT-4o-mini chỉ tiêu tốn $6.30/ngày ($0.0006/1K tokens). Như vậy, GPT-4o đắt gấp **16.67 lần** (khoảng 16.7 lần) so với GPT-4o-mini.

**Mô tả một trường hợp mà chi phí cao hơn của GPT-4o là xứng đáng, và một trường hợp GPT-4o-mini là lựa chọn tốt hơn:**
> - **GPT-4o xứng đáng:** Khi cần thực hiện các tác vụ đòi hỏi tư duy logic phức tạp, giải quyết bài toán đa bước hoặc phân tích tài liệu/dữ liệu chuyên sâu cần độ chính xác cao (ví dụ: Trợ lý phân tích luật pháp hoặc chẩn đoán y tế).
> - **GPT-4o-mini tốt hơn:** Khi cần chạy các tác vụ đơn giản, lặp đi lặp lại như phân loại ý định người dùng (intent classification), tóm tắt tin nhắn ngắn, trích xuất thông tin có cấu trúc (entity extraction) trên quy mô lớn để tối ưu chi phí và tốc độ.

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các giao diện trò chuyện thời gian thực (real-time chat, trợ lý ảo) nhằm giảm thiểu "độ trễ cảm nhận" (perceived latency) của người dùng bằng cách hiển thị ngay câu trả lời khi mô hình vừa sinh ra token đầu tiên. Ngược lại, non-streaming phù hợp hơn khi thực hiện các nhiệm vụ xử lý bất tuần tự hoặc bất đồng bộ dưới nền (background/batch tasks), các API dịch vụ trả về định dạng dữ liệu có cấu trúc (như JSON) cho các hệ thống phần mềm khác tiêu thụ toàn bộ kết quả một lần.


## Phần 3 — Giao Diện Trực Quan Với Gradio

Tôi đã xây dựng một file giao diện trực quan tại [gradio_app.py](file:///d:/Work/project/Day01-lab-assignment/solution/gradio_app.py).

### Tính Năng Giao Diện:
1. **Chatbot Streaming**: Hiển thị phản hồi từ mô hình ngay khi các token đầu tiên được tạo ra, mang lại trải nghiệm thời gian thực tuyệt vời.
2. **Chọn Mô Hình Động**: Dropdown cho phép chuyển đổi linh hoạt giữa các mô hình như `nvidia/minimaxai/minimax-m2.7`, `kc/kilo-auto/free`, và `test-combo`.
3. **Cấu Hình Tham Số**: Thanh trượt điều chỉnh trực quan giá trị Temperature (0.0 – 2.0) và Max Tokens (64 – 2048).
4. **Lịch Sử Hội Thoại**: Tự động duy trì ngữ cảnh tối đa 3 lượt hội thoại gần nhất.
5. **Giao Diện Premium**: Sử dụng Theme Soft với tông màu tối thời thượng (purple/indigo/slate).

## Danh Sách Kiểm Tra Nộp Bài
- [x] Tất cả tests pass: `pytest tests/ -v`
- [x] `call_openai` đã triển khai và kiểm thử
- [x] `call_openai_mini` đã triển khai và kiểm thử
- [x] `compare_models` đã triển khai và kiểm thử
- [x] `streaming_chatbot` đã triển khai và kiểm thử
- [x] `retry_with_backoff` đã triển khai và kiểm thử
- [x] `batch_compare` đã triển khai và kiểm thử
- [x] `format_comparison_table` đã triển khai và kiểm thử
- [x] `exercises.md` đã điền đầy đủ
- [x] Sao chép bài làm vào folder `solution` và đặt tên theo quy định 
