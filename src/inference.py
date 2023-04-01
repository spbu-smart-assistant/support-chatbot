from librosa import load, resample
import soundfile as sf
import nemo.collections.asr as nemo_asr

# init model
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
    model_name="stt_ru_conformer_ctc_large")

# set correct path
audio_path = '/content/Audio1.mp3'

# read, resample and write in correct extension
audio, sample_rate = load(audio_path)
if sample_rate != 16000:
    audio = resample(audio, orig_sr=sample_rate, target_sr=16000)
sf.write(file='/content/Audio1.wav', data = audio,
         samplerate=16000, format='wav')

# transcribe
audio_path = '/content/Audio1.wav'
transcription = asr_model.transcribe([audio_path])

print(transcription)
