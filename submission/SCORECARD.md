# Provisional rubric scorecard

**Sinh viên:** Nguyễn Kỳ Anh — **MSSV:** 2A202601558

**Ngày audit:** 24/08/2026

**Điểm ước tính nghiêm ngặt theo bằng chứng hiện có:** **77/100 core, 0 bonus**

Đây là tự chấm theo `rubric.md`, không phải điểm chính thức của giảng viên.

| Tiêu chí core | Tối đa | Hiện tại | Bằng chứng / lý do trừ |
|---|---:|---:|---|
| SFT adapter có `r=16`, `lora_alpha=32` | 6 | 6 | `adapters/sft-mini/adapter_config.json` |
| SFT loss giảm trong một epoch | 6 | 6 | `02-sft-loss.png` có xu hướng giảm khoảng 1,97 → 1,26 |
| Sample generation được in trong NB1 | 5 | 0 | Không có executed notebook giữ output NB1 |
| Parquet có `prompt/chosen/rejected` | 6 | 6 | 2.000 dòng, đúng ba cột |
| Ba ví dụ được in và `chosen != rejected` | 6 | 0 | Dữ liệu đã audit nhưng không có output cell NB2; xem `ARTIFACT_AUDIT.md` |
| DPO adapter riêng | 6 | 6 | SFT/DPO checkpoint có SHA-256 khác nhau |
| Reward gap tăng đúng chiều | 12 | 0 | Gap cuối `−1,4715`, rejected reward cao hơn chosen |
| Vẽ và diễn giải cả hai reward | 10 | 10 | `03-dpo-reward-curves.png` + Reflection §3 |
| Bảng ≥8 prompt × 2 output | 8 | 8 | `04-side-by-side-table.png` + JSONL 8 dòng |
| Summary, 4 helpfulness + 4 safety | 7 | 7 | DPO thắng 2, hòa 6, thua 0 |
| Tái lập bằng pipeline/Run-all | 5 | 5 | Notebook one-click + 5/5 smoke tests |
| Reflection §3 và §6 ≥150 từ | 15 | 15 | Lần lượt khoảng 260 và 260 token tách theo whitespace |
| §3 diễn giải chosen/rejected riêng | 5 | 5 | Có số cuối kỳ và chẩn đoán đúng failure mode |
| `make verify` / `verify.py` exit 0 | 3 | 3 | Core gate qua, reward âm được cảnh báo non-blocking |
| **Tổng core** | **100** | **77** | |

## Cách đạt 100 core

1. Chạy lại notebook Colab từ commit `981952a`; lần chạy mới phải dùng base/reference
   của notebook mới và có reward gap dương. Nếu đạt, tiêu chí reward gap có thể lấy lại 12 điểm.
2. Tải notebook `.ipynb` **sau khi chạy xong**, giữ toàn bộ output cells. NB1 phải hiện
   ít nhất một sample generation và NB2 phải hiện ba cặp preference; hai mục này có thể lấy lại 11 điểm.
3. Chụp thêm `01-setup-gpu.png`. Ảnh judge đã được dựng từ verdict thật thành
   `05-judge-output.png`.

## Bonus chưa có

- NB5 GGUF + llama.cpp smoke: +6.
- NB6 benchmark + Reflection §7 bằng số liệu thật: +8.
- β-sweep `{0.05, 0.1, 0.5}`: +6.

Ba mục trên vừa đủ chạm trần +20 nếu được chạy và phân tích đầy đủ. Ảnh
`06-gguf-smoke.png`, `07-benchmark-comparison.png` và `bonus-beta-sweep.png` không thể
tạo hợp lệ từ ZIP hiện tại vì không có output tương ứng.
