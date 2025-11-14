### Bai2 Speech Processing

# Yêu cầu môi trường
- Python 3.10 hoặc 3.11

# Thư viện Python
- pydub
- librosa
- soundfile
- jiwer
- tqdm
- transformers
- torch (CUDA hoặc CPU)
- torchaudio

# Cài đặt
```
pip install pydub librosa soundfile jiwer tqdm transformers torch torchaudio
```

# FFmpeg
Bắt buộc để đọc và xử lý file MP3.

Kiểm tra phiên bản:
```
ffmpeg -version
ffprobe -version
```

## Cách chạy chương trình
### Bước 1 – Activate virtual environment
```
./venv/Scripts/activate
```

### Bước 2 – Chạy pipeline
Truy cập thư mục:
```
voice_processing_pipeline/Bai2_voice_processing
```

Chạy lệnh:
```
python ex2_pipeline.py
```