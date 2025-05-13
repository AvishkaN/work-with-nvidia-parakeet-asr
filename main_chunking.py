import os
import torch
import torchaudio
import tempfile
from more_itertools import chunked
from tqdm import tqdm
from nemo.collections.asr.models import EncDecRNNTBPEModel

# Parameters
AUDIO_FILE = "Best Motivational Speech Compilation EVER #33 - DETERMINED ｜ 1 Hour of the Best Motivation.wav"
CHUNK_DURATION_SEC = 10
BATCH_SIZE = 10
SAMPLE_RATE = 16000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load ASR model
asr_model = EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2").to(device)

def load_audio_chunk(file_path, offset, num_frames):
    audio, sr = torchaudio.load(file_path, frame_offset=offset, num_frames=num_frames)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        audio = resampler(audio)
    return audio

def chunk_audio(file_path, chunk_duration_sec, sample_rate):
    info = torchaudio.info(file_path)
    total_frames = info.num_frames
    chunk_size = chunk_duration_sec * sample_rate
    chunks = []

    for offset in range(0, total_frames, chunk_size):
        num_frames = min(chunk_size, total_frames - offset)
        chunks.append((offset, num_frames))

    return chunks

def save_chunk_to_wav(audio_tensor, sample_rate):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    torchaudio.save(temp_file.name, audio_tensor, sample_rate)
    return temp_file.name

def main():
    print(f"Preprocessing: {AUDIO_FILE}")
    chunks = chunk_audio(AUDIO_FILE, CHUNK_DURATION_SEC, SAMPLE_RATE)
    print(f"Split into {len(chunks)} chunks. Starting transcription...")
    print(f"Transcribing in batches of {BATCH_SIZE}...")

    full_transcription = []
    temp_files = []

    for batch in tqdm(chunked(chunks, BATCH_SIZE)):
        file_batch = []
        for offset, num_frames in batch:
            try:
                audio_tensor = load_audio_chunk(AUDIO_FILE, offset, num_frames)
                file_path = save_chunk_to_wav(audio_tensor, SAMPLE_RATE)
                temp_files.append(file_path)
                file_batch.append(file_path)
            except Exception as e:
                print(f"[ERROR] Failed to process chunk: {e}")

        try:
            with torch.no_grad():
                transcripts = asr_model.transcribe(paths2audio_files=file_batch)
            full_transcription.extend(transcripts)
        except Exception as e:
            print(f"[ERROR] Batch failed: {e}")
            torch.cuda.empty_cache()

    print("\nFull Transcription:\n")
    print("\n".join(full_transcription))

    # Cleanup temp files
    for f in temp_files:
        os.remove(f)

if __name__ == "__main__":
    main()
