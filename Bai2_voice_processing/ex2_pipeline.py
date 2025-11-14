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
from torch.utils.data import Dataset, DataLoader
import torchaudio


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


# =====================================================
#                STEP 4 – BUCKETED DATALOADERS (FIXED)
# =====================================================

# =====================================================
#                STEP 4 – BUCKETED DATALOADERS (NO WARNING)
# =====================================================

class AudioBucketDataset(Dataset):
    def __init__(self, data_list, processor):
        self.data_list = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        wav_path, transcript = self.data_list[idx]

        # Load audio as numpy array
        waveform, sr = librosa.load(wav_path, sr=16000)

        return {
            "waveform": waveform,   # numpy array
            "transcript": transcript
        }


def create_collate_fn(processor):
    def collate_fn(batch):

        # 1) List audio arrays
        waveforms = [item["waveform"] for item in batch]
        # 2) List transcripts
        texts = [item["transcript"] for item in batch]

        # ----- AUDIO PROCESSING -----
        audio_inputs = processor(
            audio=waveforms,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt"
        )

        # ----- TEXT PROCESSING (new API: processor(text=...) ) -----
        text_inputs = processor(
            text=texts,
            padding=True,
            return_tensors="pt"
        )

        # Replace PAD by -100 for CTC loss
        labels = text_inputs.input_ids
        labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)

        audio_inputs["labels"] = labels
        return audio_inputs

    return collate_fn


def step4_create_dataloaders():
    print("\n--- Step 4: Creating Bucketed DataLoaders ---")

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    test_map = load_test_txt()

    group_short, group_long = [], []

    for wav_file in tqdm(WAV_DIR.glob("*.wav"), desc="Grouping"):
        info = sf.info(str(wav_file))
        dur = info.frames / info.samplerate

        fid = wav_file.stem
        transcript = test_map.get(fid, "").lower().strip()

        if dur <= 4.0:
            group_short.append((str(wav_file), transcript))
        else:
            group_long.append((str(wav_file), transcript))

    print(f"Nhóm ngắn (<=4s): {len(group_short)}")
    print(f"Nhóm dài  (>4s): {len(group_long)}")

    dataset_short = AudioBucketDataset(group_short, processor)
    dataset_long = AudioBucketDataset(group_long, processor)

    collate_fn = create_collate_fn(processor)

    loader_short = DataLoader(dataset_short, batch_size=6, shuffle=True, collate_fn=collate_fn)
    loader_long = DataLoader(dataset_long, batch_size=4, shuffle=True, collate_fn=collate_fn)

    print("\n=== Mechanism Created ===")
    print(f"→ Batch ngắn = {loader_short.batch_size}")
    print(f"→ Batch dài  = {loader_long.batch_size}")

    return loader_short, loader_long



# ----------------- MAIN -----------------
if __name__ == "__main__":
    step1_get_info_and_convert()
    step3_asr_and_metrics()
    
    loader_short, loader_long = step4_create_dataloaders()

    print("\n--- Demo: Loading 1 batch ---")

    try:
        b = next(iter(loader_short))
        print("Short batch OK:", b["input_values"].shape, b["labels"].shape)
    except Exception as e:
        print("Short batch ERROR:", e)

    try:
        b = next(iter(loader_long))
        print("Long batch OK:", b["input_values"].shape, b["labels"].shape)
    except Exception as e:
        print("Long batch ERROR:", e)
