import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
import time


startTime=time.time()
transcriptions = asr_model.transcribe(["download (6).wav"])
endTime=time.time()
print(f'time taken {endTime-startTime}')
print(transcriptions[0])