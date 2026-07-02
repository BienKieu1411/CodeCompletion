# Code Completion - Co-Retrieval Framework

Repository này chứa mã nguồn triển khai framework huấn luyện `co_retrieval` (thay thế cho phiên bản `graphcoder_rl` trước đó). Framework này tập trung vào các kỹ thuật truy xuất ngữ cảnh nâng cao cho bài toán tự động hoàn thiện mã nguồn (code completion), tận dụng cây cú pháp trừu tượng (AST) và các luồng huấn luyện neural.

## Các tính năng và Cập nhật gần đây

Dựa trên các thay đổi gần nhất của dự án, các tính năng và cải tiến kiến trúc chính sau đã được triển khai:

### 1. Framework `co_retrieval` Mới
- **Tái cấu trúc kiến trúc:** Thay thế mã nguồn `graphcoder_rl` cũ bằng một framework huấn luyện `co_retrieval` được thiết kế mới nhằm hỗ trợ tốt hơn các cơ chế truy xuất ngữ cảnh nâng cao và nhiều bước (multi-step).

### 2. Pipeline Huấn luyện Neural
- **DPO-based Dense Retriever:** Xây dựng pipeline huấn luyện neural hoàn chỉnh với bộ truy xuất dày đặc (dense retriever) được huấn luyện bằng phương pháp Direct Preference Optimization (DPO).
- **Soft Prompting & Neural Gates:** Tích hợp cơ chế soft prompting và cổng neural (neural gates) để tự động kiểm soát, định tuyến và tối ưu hóa quá trình truy xuất ngữ cảnh.
- **Đánh giá Gate:** Thêm các số liệu (metrics) toàn diện để đo lường và phân tích hiệu suất của các cơ chế cổng neural.

### 3. Các Chiến lược Truy xuất Ngữ cảnh
- **Truy xuất theo Intent:** Triển khai truy xuất ngữ cảnh dựa trên ý định (intent-based) nhằm lấy ra các đoạn mã liên quan nhất dựa trên ý nghĩa (semantic) của câu truy vấn.
- **Truy xuất Tuần tự:** Hỗ trợ các chiến lược truy xuất tuần tự (sequential retrieval) để thu thập và tinh chỉnh ngữ cảnh qua nhiều bước.
- **Xử lý Dữ liệu:** Xây dựng cấu trúc phân chia dữ liệu huấn luyện (training splits) và các tiện ích hỗ trợ được thiết kế riêng cho luồng xử lý `co_retrieval`.

### 4. Thư viện Phân mảnh Code (Code Chunking) dựa trên AST
- **Phân mảnh theo cấu trúc (Structural Chunking):** Xây dựng thư viện `astchunk` chuyên biệt để phân tích và chia nhỏ mã nguồn một cách logic dựa trên cú pháp ngôn ngữ.
- **Ranh giới Tùy chỉnh:** Hỗ trợ tùy chỉnh ranh giới phân chia cấu trúc (configurable boundaries), cho phép phân đoạn mã nguồn linh hoạt và chính xác.
- **Hỗ trợ Metadata:** Tích hợp dữ liệu siêu meta (metadata) phong phú cho các đoạn code AST đã phân tích, hỗ trợ bộ truy xuất dày đặc (dense retriever) trong việc xác định và xếp hạng các ngữ cảnh.

---
*Lưu ý: README này tổng hợp các tiến trình chuyển đổi sang framework `co_retrieval` gần đây. Hướng dẫn cài đặt, thiết lập môi trường và cách sử dụng chi tiết sẽ được bổ sung sau.*
