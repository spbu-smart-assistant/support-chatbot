from fastapi import FastAPI, UploadFile, File
import shutil
import nemo.collections.asr as nemo_asr
import uvicorn
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from librosa import load, resample
import soundfile as sf
import concurrent.futures

app = FastAPI()

MODEL_NAME = 'cointegrated/rut5-base-absum'
summa_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
summa_model.cuda() if torch.cuda.is_available() else None
summa_model.eval()
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name="stt_ru_conformer_ctc_large"
    )

AUDIO_FOLDER = "../FastAPI/audio/"

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

def process_transcription(input_path):
    try:

        # read, resample and write in correct extension
        audio, sample_rate = load(input_path)
        if sample_rate != 16000:
            audio = resample(audio, orig_sr=sample_rate, target_sr=16000)
        sf.write(file='../src/Audio1_1.wav', data=audio,
                 samplerate=16000, format='wav')

        # transcribe
        audio_path = '../src/Audio1_1.wav'
        transcription = asr_model.transcribe([audio_path])

        # summarize
        summarization = summarize(transcription)
        return summarization
    except FileNotFoundError:
        return "Incorrect file name or the file is corrupted"

@app.get("/process/{input_path}")
async def process_string(input_path: str):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(process_transcription, input_path)
        summarization = future.result()

    return summarization

@app.post("/file")
async def handle_audio(file: UploadFile = File()):
    with open(f"{AUDIO_FOLDER}{file.filename}", 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"file_name": file.filename}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)