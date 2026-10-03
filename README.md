# OmniVoice Studio

**Developed by Kenzy.**

Giao diện web chạy local cho [OmniVoice](https://github.com/k2-fsa/OmniVoice) (k2-fsa, Apache-2.0):
nhân bản giọng từ 3–10 giây audio, thiết kế giọng bằng thuộc tính, hoặc để model tự chọn giọng.
Chạy trên GPU của bạn, không giới hạn ký tự, không gửi dữ liệu đi đâu.

## Chạy

Bấm đúp `start.bat`, hoặc:

```bash
.venv/Scripts/python.exe server.py
```

Mở <http://127.0.0.1:8765>. Model và Whisper nạp sẵn lúc khởi động (~30–60 s lần đầu).
Tuỳ chọn: `--port 9000`, `--no-preload`, `--no-asr`, `--model vi`. Trong Claude Code: cấu hình `studio` trong `.claude/launch.json`.

## Tính năng

- **Một lần**: một ô text → một file. **Theo dòng**: dán cả kịch bản (hoặc `script.txt` dạng `b01: … [sfx: @ ]`, app tự bỏ nhãn,
  dòng `HOOK:`/`THUMB…`, chỗ trống SFX và dấu `*`), mỗi dòng tạo riêng, thu lại riêng dòng hỏng (seed khác), chọn sắc thái theo dòng,
  **Xuất** ra `data/projects/<id>/export/`: `bNN.wav` + `bNN.mp3` + `bNN.json` (alignment từng ký tự kiểu ElevenLabs, dùng thẳng
  cho `recap-shorts/public/voice/<ep>/`), `full.wav`/`full.mp3` ghép có khoảng nghỉ, `timeline.json` (mốc từng dòng và từng chữ), `export.zip`.
- **Soát lỗi đọc bằng Whisper**: sau khi tạo, Whisper nghe lại và so với lời; chữ đỏ = đọc sai/bỏ sót, vàng = gần đúng (thường chỉ là
  cách Whisper viết: sét/xét, ổng/ông). **Số bản thử** 1–5: tạo tới khi có bản sạch lỗi, giữ bản ít lỗi nhất.
- **Từ điển phát âm** (`data/pronunciation.json`): Enel → Ê-nen, vcl → vê-cờ-lờ…; text hiển thị giữ nguyên, chỉ phần đưa cho model đổi.
  **Đọc số thành chữ**: 100 → một trăm, 3,5 → ba phẩy năm, 50% → năm mươi phần trăm. Bật "Xem lời sẽ đọc" để kiểm tra.
- **Nhiều sắc thái cho một giọng**: OmniVoice bắt chước phong cách của đoạn mẫu, nên một giọng có thể giữ nhiều mẫu (bình thường, bựa, hét…)
  và chọn theo từng dòng. Bấm "+ Sắc thái" ở kết quả để lưu một bản vừa tạo làm sắc thái.
- **Tốc độ**: `speed` của OmniVoice là ép thời lượng — ngoài 0,85–1,2× nó ngắt quãng (chậm) hoặc nuốt chữ (nhanh). Chế độ "Tự động" giữ
  model trong khoảng đó và kéo giãn phần còn lại bằng rubberband (giữ nguyên chữ và cao độ). Đổi được trong Nâng cao.
- **Căn chữ**: mốc thời gian từng chữ (Whisper) cho caption, tải `.json` ở kết quả hoặc có sẵn trong file xuất.
- Thêm giọng: kéo file hoặc ghi âm micro; file dài hơn 20 s được cắt ở chỗ ngắt hơi gần 10 s và Whisper chép lại lời; khoảng lặng đầu/cuối
  tự cắt; cảnh báo khi mẫu có nhạc nền/ồn (SNR < 25 dB).

## Cài lại từ đầu

```bash
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install omnivoice fastapi "uvicorn[standard]" python-multipart num2words
```

Cần `ffmpeg` trên PATH (có filter `rubberband` thì tốt, không thì dùng `atempo`). Model tải về `~/.cache/huggingface/hub`
(`k2-fsa/OmniVoice` ~3,1 GB; `splendor1811/omnivoice-vietnamese` ~2,3 GB; Whisper large-v3-turbo ~1,6 GB).

## Cấu trúc

- `server.py` — FastAPI; tài liệu API tại `/api/docs`.
- `static/` — giao diện (HTML/CSS/JS thuần, font tự host).
- `data/voices/<id>/` — `meta.json` (danh sách sắc thái), `ref.wav`, `styles/<sid>.wav`, `prompt-<model>-<sid>.pt` (cache).
  Sáu giọng `preset-*` lấy từ bộ mẫu của VieNeu-TTS (Apache-2.0).
- `data/outputs/<id>.{wav,json}` — lịch sử; json có `spoken_text`, `qa`, `alignment`, `takes`.
- `data/projects/<id>/` — dự án theo dòng và thư mục `export/`.

## Lưu ý

- Model fine-tune tiếng Việt khi đo hay nuốt chữ và dùng dữ liệu CC-BY-NC-SA → để "thử nghiệm"; model gốc là mặc định.
- Chỉ nhân bản giọng của bạn hoặc giọng bạn có quyền dùng. Ghi "Giọng đọc AI" khi đăng.
- Lời mẫu phải là **đúng những gì có trong đoạn mẫu**; dán cả kịch bản dài hơn file sẽ làm model đọc sai (nhặt chữ từ phần chưa nói).
- Đo trên RTX 5060 8 GB: ~10 s audio trong ~1,8 s, cộng ~1 s soát lỗi; VRAM ~3,5 GB có cả Whisper.
