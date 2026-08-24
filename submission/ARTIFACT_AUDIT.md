# Artifact audit — Lab 22

**Sinh viên:** Nguyễn Kỳ Anh
**MSSV:** 2A202601558

## Sources audited

- `lab22-results.zip` — SHA-256 `37898568CE6CDD44E35CADC0052F3013C332DB0588A9F874D13427BD887C0E26`
- Executed notebook `colab/Lab22_DPO_T4_SAFE.ipynb` — SHA-256 `A14D327474B0123F96C4499E6D6AD6CCFACE43B1B5C92A3DC68CF0E65EA393DD`

## Valid artifacts

| Hạng mục | Kết quả kiểm tra |
|---|---|
| Notebook | 45/45 code cells đã chạy; 0 output kiểu `error`; final verify pass |
| GPU | Tesla T4 14,6 GiB |
| SFT config | `r=16`, `lora_alpha=32`, base `unsloth/Qwen2.5-3B-bnb-4bit` |
| SFT sanity | Có prompt và response thật trong NB1 |
| DPO config | Có tại đường dẫn riêng `adapters/dpo/adapter_config.json` |
| DPO metrics | loss `0,6738`; chosen `−0,0177`; rejected `−0,0863`; gap `+0,0686` |
| Preference data | 2.000 dòng; đúng `prompt/chosen/rejected`; ba examples được in trong NB2 |
| Eval | 8 cặp output; 4 helpfulness + 4 safety |
| Manual rubric | DPO thắng 2; SFT thắng 3; hòa 3 |
| Screenshot | SFT loss, DPO reward curves, side-by-side, manual verdicts |

Reward plot thể hiện gap tăng từ gần 0 lên vùng dương và vẽ riêng chosen/rejected. SFT loss giảm tổng thể từ khoảng 1,56 xuống khoảng 1,13 dù có dao động mini-batch.

## Ba preference examples

Notebook đã in trực tiếp ba cặp sau và Parquet xác nhận `chosen != rejected`:

1. C++ kiểm tra quốc gia giáp Địa Trung Hải.
2. Tự động tạo tiêu đề/mô tả YouTube bằng GPT.
3. Phân tích khủng hoảng chứng khoán năm 1929.

## Warnings and scope

- Các dòng `Exception ignored ... BufferError` là cảnh báo tiến trình phụ của Python 3.13/datasets; không có output notebook kiểu `error`, và các bước sau đều hoàn thành.
- Không có `adapter_model.safetensors`; submission theo Option C/code-only, vẫn đủ core.
- Không có GGUF, benchmark, β-sweep, Hugging Face/W&B link hoặc cross-judge; vì vậy bonus hiện là 0/20.
