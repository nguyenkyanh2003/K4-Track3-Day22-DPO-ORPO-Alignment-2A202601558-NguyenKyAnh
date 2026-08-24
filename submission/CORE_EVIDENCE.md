# Core evidence index

Notebook chính đã chạy và giữ output:
`colab/Lab22_DPO_T4_SAFE.ipynb`.

- SHA-256: `A14D327474B0123F96C4499E6D6AD6CCFACE43B1B5C92A3DC68CF0E65EA393DD`
- Kernel: Python 3; accelerator metadata: GPU.
- 45/45 code cells có `execution_count`.
- 0 output có kiểu `error`.
- GPU output: Tesla T4, 14,6 GiB.
- Phiên bản chạy: Transformers 4.57.6, TRL 0.19.1, PEFT 0.20.0, Unsloth 2026.4.4.

## Rubric evidence

- **NB1:** code cell 16 in prompt và `SFT-mini response`; loss cuối `1,1547`; adapter được lưu thành công.
- **NB2:** code cell 21 in Example 1–3 với prompt/chosen/rejected; Parquet có 2.000 cặp.
- **NB3:** DPO loss cuối `0,6738`; chosen `−0,018`, rejected `−0,086`, gap `+0,069`; adapter và metrics được lưu.
- **NB4:** sinh đủ tám response cho từng adapter; in bảng 8 prompt × 2 model và win/loss/tie.
- **Final:** output `Core checks passed` và `READY TO SUBMIT`.

Một số cell có cảnh báo `BufferError: Existing exports of data` từ tiến trình phụ của Python 3.13/datasets. Đây là stderr dạng `Exception ignored`, không phải notebook error; pipeline vẫn huấn luyện, lưu artifact và verify thành công.
