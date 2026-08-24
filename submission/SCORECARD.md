# Provisional rubric scorecard

**Sinh viên:** Nguyễn Kỳ Anh — **MSSV:** 2A202601558
**Ngày audit:** 24/08/2026

**Điểm ước tính nghiêm ngặt theo artifact hiện có: 100/100 core, 0/20 bonus.**

Đây là tự chấm theo `rubric.md`, không phải điểm chính thức của giảng viên. Mọi điểm bên dưới đều có artifact hoặc output chạy thật; không dùng placeholder.

| Tiêu chí core | Max | Ước tính | Bằng chứng |
|---|---:|---:|---|
| SFT config `r=16`, `lora_alpha=32` | 6 | 6 | `adapters/sft-mini/adapter_config.json` |
| SFT loss giảm tổng thể | 6 | 6 | `submission/screenshots/02-sft-loss.png` |
| Sample generation in trong NB1 | 5 | 5 | Executed notebook, code cell 16: `SFT-mini response` |
| Parquet đúng ba cột | 6 | 6 | `data/pref/train.parquet`: 2.000 dòng |
| Ba examples in trong NB2 | 6 | 6 | Executed notebook, code cell 21: Example 1–3, mỗi cặp `chosen != rejected` |
| DPO adapter config riêng | 6 | 6 | `adapters/dpo/adapter_config.json` |
| Reward gap tăng đúng chiều | 12 | 12 | Gap cuối `+0,0686`; plot tăng từ gần 0 |
| Hai reward curves + diễn giải | 10 | 10 | Plot + `submission/REFLECTION.md` §3 |
| Bảng 8 prompt × 2 output | 8 | 8 | JSONL + screenshot + executed notebook |
| Summary 4 helpfulness + 4 safety | 7 | 7 | Manual audit: DPO 2, SFT 3, hòa 3 |
| Tái lập bằng Run-all | 5 | 5 | 45/45 code cells đã chạy; final verify thành công |
| Reflection §3 và §6 ≥150 từ | 15 | 15 | `submission/REFLECTION.md` |
| §3 đọc riêng chosen/rejected | 5 | 5 | Có số liệu và chẩn đoán likelihood displacement |
| `verify.py` exit 0 | 3 | 3 | Chạy lại sau khi cấu trúc submission |
| **Tổng core** | **100** | **100** | |

## Bonus

| Add-on | Max | Hiện tại | Lý do |
|---|---:|---:|---|
| NB5 GGUF deploy | 6 | 0 | Không có `.gguf` hoặc smoke response |
| NB6 benchmark | 8 | 0 | Không có benchmark JSON/plot |
| β-sweep | 6 | 0 | Chỉ có β = 0,1 |
| Hugging Face push | 5 | 0 | Không có Hub URL/model card |
| GGUF multi-quant release | 3 | 0 | Không có release |
| MMLU full | 3 | 0 | Chưa chạy NB6 |
| W&B | 2 | 0 | Không có public run URL |
| Cross-judge | 4 | 0 | Chỉ có manual rubric |

Bonus không thể được cấp bằng placeholder. Các mục này bắt buộc có kết quả GPU/API hoặc URL public thật.
