# Artifact audit — Lab 22

**Sinh viên:** Nguyễn Kỳ Anh

**MSSV:** 2A202601558

**Nguồn kiểm tra:** `lab22-submission.zip`, ngày 24/08/2026

## Kết quả kiểm tra tự động

| Hạng mục | Kết quả xác minh |
|---|---|
| SFT adapter | `r=16`, `lora_alpha=32`, base `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` |
| DPO adapter | Có checkpoint riêng; SHA-256 khác SFT adapter |
| Preference data | 2.000 dòng, đúng ba cột `prompt`, `chosen`, `rejected` |
| Chất lượng cặp preference | 0/2.000 dòng có `chosen == rejected` |
| DPO cuối kỳ | chosen `+1.0752`, rejected `+2.5467`, gap `−1.4715`, loss `2.1202` |
| So sánh định tính | 8 prompt: 4 helpfulness + 4 safety |
| Judge | DPO thắng 2, hòa 6, SFT thắng 0 |
| Đầu ra giống hệt nhau | 6/8 cặp SFT và DPO |

SHA-256 của hai checkpoint LoRA:

- SFT: `BF87119A6506EED020414489E5B09B080696A487BA82EB7BA567C86F9699A3C3`
- DPO: `1CC62D85C89E772ADD94B5D05E980DD2DE575AA9F7534D0A32E0CA5A190785B9`

## Ba cặp preference đã kiểm tra

1. **Prompt:** viết chương trình C++ kiểm tra một quốc gia có giáp Địa Trung Hải hay không. **Chosen:** đưa ra chương trình dùng cấu trúc dữ liệu ánh xạ quốc gia/biển. **Rejected:** dùng API/biểu thức C++ không hợp lệ. Hai câu trả lời khác nhau.
2. **Prompt:** trình bày cách dùng GPT để tự động tạo tiêu đề và mô tả YouTube. **Chosen:** nêu tuần tự chọn model, lấy API key và xây pipeline. **Rejected:** giả định phải tự huấn luyện GPT nhưng thiếu hướng dẫn triển khai chính xác. Hai câu trả lời khác nhau.
3. **Prompt:** phân tích các yếu tố kinh tế, chính trị và xã hội dẫn đến khủng hoảng chứng khoán 1929. **Chosen:** đi thẳng vào các yếu tố liên quan và cấu trúc bài phân tích. **Rejected:** mở đầu chung và ít bám sát yêu cầu so sánh hơn. Hai câu trả lời khác nhau.

## Giới hạn của bộ artifact hiện tại

- Không có notebook Colab **đã chạy và giữ output cells**, nên chưa có bằng chứng trực tiếp cho sample generation ở NB1 và ba ví dụ được in ở NB2.
- Không có ảnh `01-setup-gpu.png`; tier T4 chỉ được xác nhận qua `dpo_metrics.json` và tiêu đề biểu đồ.
- Không có GGUF, llama.cpp smoke test, benchmark hoặc beta sweep.
- Reward gap âm cho thấy lần DPO này chưa học đúng chiều preference. Báo cáo giữ nguyên kết quả thật; không chỉnh sửa số liệu hoặc biểu đồ.
