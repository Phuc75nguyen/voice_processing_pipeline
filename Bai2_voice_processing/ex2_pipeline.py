import os
from pathlib import Path
from pydub import AudioSegment
import soundfile as sf
import librosa
from jiwer import wer
from tqdm import tqdm
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# ===================== CONFIG ======================

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR / "common_voice"
MP3_DIR = ROOT / "mp3"
WAV_DIR = ROOT / "wav16k"
TEST_TXT = ROOT / "test.txt"
EX2_1_OUT = ROOT / "ex2_1.txt"
MODEL_NAME = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# đảm bảo thư mục wav16k tồn tại
WAV_DIR.mkdir(parents=True, exist_ok=True)
# ====================================================

# ----------------- STEP 1 + 2 -----------------
def step1_get_info_and_convert():
    rows = [] # list of tuples (file_id, sample_rate, duration)
    mp3_files = sorted(MP3_DIR.glob("*.mp3")) 

    for mp3_file in tqdm(mp3_files, desc="Processing mp3"):
        audio = AudioSegment.from_file(mp3_file) # load mp3
        sr = audio.frame_rate # sample rate
        dur = len(audio) / 1000.0 # duration in seconds
        rows.append((mp3_file.stem, sr, dur)) # collect info

        # convert to wav (16k Hz mono)
        wav_path = WAV_DIR / f"{mp3_file.stem}.wav"  # output path
        audio = audio.set_frame_rate(16000).set_channels(1)  # resample to 16kHz mono
        audio.export(wav_path, format="wav") # export as wav

    # Write ex2_1.txt
    with open(EX2_1_OUT, "w", encoding="utf-8") as f:
        for file_id, sr, dur in rows:
            f.write(f"{file_id}\t{sr}\t{dur:.3f}\n")

    print(f"Step 1 & 2 done → {EX2_1_OUT}")


# ----------------- STEP 3: ASR + METRICS -----------------
def load_test_txt():
    mapping = {}
    with open(TEST_TXT, "r", encoding="utf-8") as f:
        for line in f:
            fid, txt = line.strip().split("\t", 1)
            mapping[fid] = txt
    return mapping

def compute_cer(ref, hyp): # character error rate
    ref = ref.replace(" ", "")
    hyp = hyp.replace(" ", "")
    if len(ref) == 0:
        return 1.0 if len(hyp) > 0 else 0

    dp = np.zeros((len(ref)+1, len(hyp)+1), dtype=np.int32)
    for i in range(len(ref)+1): dp[i][0] = i
    for j in range(len(hyp)+1): dp[0][j] = j

    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[len(ref)][len(hyp)] / len(ref)

def step3_asr_and_metrics(): # ASR + WER, CER
    print("Loading ASR model…")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    test_map = load_test_txt()
    wav_files = sorted(WAV_DIR.glob("*.wav"))

    refs, hyps = [], []

    for wav_file in tqdm(wav_files, desc="Transcribing"):
        fid = wav_file.stem

        audio, sr = librosa.load(wav_file, sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = model(inputs.input_values.to(DEVICE)).logits

        pred_ids = torch.argmax(logits, dim=-1)
        hyp = processor.batch_decode(pred_ids)[0].lower().strip()
        ref = test_map.get(fid, "").lower().strip()

        refs.append(ref)
        hyps.append(hyp)

    global_wer = wer(refs, hyps)
    cer_scores = [compute_cer(r, h) for r, h in zip(refs, hyps)]
    global_cer = float(np.mean(cer_scores))

    print(f"WER = {global_wer:.4f}")
    print(f"CER = {global_cer:.4f}")


# ----------------- STEP 4: GROUP BY DURATION -----------------
def step4_grouping():
    short, long = [], []

    for wav_file in WAV_DIR.glob("*.wav"):
        info = sf.info(str(wav_file))
        dur = info.frames / info.samplerate
        if dur <= 4.0:
            short.append(wav_file.name)
        else:
            long.append(wav_file.name)

    print("\n=== GROUP RESULT ===")
    print("Short (<=4s):", len(short))
    print("Long  (>4s):", len(long))
    print("Batch Short = 6")
    print("Batch Long  = 4")


# ----------------- MAIN -----------------
if __name__ == "__main__":
    step1_get_info_and_convert()
    step3_asr_and_metrics()
    step4_grouping()
