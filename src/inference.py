import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from librosa import load, resample
import soundfile as sf
import nemo.collections.asr as nemo_asr

# init summa model an tokenizer
MODEL_NAME = 'cointegrated/rut5-base-absum'
summa_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
summa_model.cuda() if torch.cuda.is_available() else None
summa_model.eval();

def summarize(
    text, n_words=None, compression=None,
    max_length=1500, num_beams=3, do_sample=False, repetition_penalty=10.0, 
    **kwargs
):
    """
    Summarize the text
    The following parameters are mutually exclusive:
    - n_words (int) is an approximate number of words to generate.
    - compression (float) is an approximate length ratio of summary and original text.
    """
    if n_words:
        text = '[{}] '.format(n_words) + text
    elif compression:
        text = '[{0:.1g}] '.format(compression) + text
    x = tokenizer(text, return_tensors='pt', padding=True).to(summa_model.device)
    with torch.inference_mode():
        out = summa_model.generate(
            **x, 
            max_length=max_length, num_beams=num_beams, 
            do_sample=do_sample, repetition_penalty=repetition_penalty, 
            **kwargs
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

# init asr model
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

# summarize
summarization = summarize(transcription)

# output
print(summarization)
