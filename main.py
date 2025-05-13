import os
import time
import torch
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment



def convert_to_mono_16k(audio_path, output_path=None):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    
    if output_path is None:
        base, ext = os.path.splitext(audio_path)
        output_path = base + "_mono.wav"
    
    audio.export(output_path, format="wav")
    return output_path


# Check if CUDA is available for GPU acceleration
device = "cuda" if torch.cuda.is_available() else "cpu"
device="cpu"
device="cuda"
print(f"Using device: {device}")

# Load pre-trained ASR model
print("Loading ASR model...")
asr_model_name="nvidia/parakeet-tdt-1.1b"
asr_model_name="nvidia/parakeet-tdt-0.6b-v2"
asr_model = nemo_asr.models.ASRModel.from_pretrained(asr_model_name)

# Move model to appropriate device
if device == "cuda":
    asr_model = asr_model.cuda()
else:
    asr_model = asr_model.cpu()



def transcribeAudio(audio_file):


    # Verify file exists
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    try:
        # Transcribe audio
        print("Transcribing audio...")
        start_time = time.time()
        transcriptions = asr_model.transcribe([audio_file])
        end_time = time.time()
        
        # Print results
        time_taken = end_time - start_time
        print(f"Time taken: {time_taken:.2f} seconds")
        print(f"Transcription: {transcriptions[0].text}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

transcribeAudio("download (7).wav")
time.sleep(5)
transcribeAudio(convert_to_mono_16k("Best Motivational Speech Compilation EVER #33 - DETERMINED ｜ 1 Hour of the Best Motivation.wav"))
# transcribeAudio("download (4).wav")
# transcribeAudio("download (2).wav")
# transcribeAudio("download (8).wav")
# transcribeAudio("download (6).wav")
# transcribeAudio("download (10).wav")
# transcribeAudio("download (11).wav")