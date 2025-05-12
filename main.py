import os
import time
import torch
import nemo.collections.asr as nemo_asr

# Check if CUDA is available for GPU acceleration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Path to audio file
audio_file = "download (6).wav"

# Verify file exists
if not os.path.exists(audio_file):
    raise FileNotFoundError(f"Audio file not found: {audio_file}")

try:
    # Load pre-trained ASR model
    print("Loading ASR model...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    
    # Move model to appropriate device
    if device == "cuda":
        asr_model = asr_model.cuda()
    
    # Transcribe audio
    print("Transcribing audio...")
    start_time = time.time()
    transcriptions = asr_model.transcribe([audio_file])
    end_time = time.time()
    
    # Print results
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken:.2f} seconds")
    print(f"Transcription: {transcriptions[0]}")
    
except Exception as e:
    print(f"An error occurred: {str(e)}")