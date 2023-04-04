"""
functions for testing models
"""

import re
import torch
import numpy as np
import soundfile as sf
from jiwer import wer, cer
from tqdm.auto import tqdm
from src.dataset_utils import read_manifest

def softmax(logits):
    """
    perfom softmax over logits from ASR model
    """

    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])

def test_transformers_asr_model(model,
                    processor,
                    message: str,
                    manifests: list,
                    ):
    """
    """
    test_text = []
    test_path = []
    for path in manifests:
        mn = read_manifest(path)
        for sample in mn:
          if sample["text"] != '':
                test_text.append(sample["text"])
                test_path.append(sample['audio_filepath'])
    transcribed_text = []
    for path in tqdm(test_path, desc='Transcribing texts...'):
        data, sample_rate = sf.read(path)
        processed = processor(data, sampling_rate=sample_rate, return_tensors='pt', padding='longest')
        logits = model(processed.input_values, attention_mask=processed.attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode([predicted_ids])[0]
        transcribed_text.append(transcription)
    try:
        WER = wer(test_text, transcribed_text)
        CER = cer(test_text, transcribed_text)
        print(f'{message}:')
        print('WER:', WER)
        print('CER:', CER, '\n')
        return test_text, transcribed_text, logits, predicted_ids
    except:
        print('Cannot calculate WER and CER')
        return test_text, transcribed_text, logits, predicted_ids

def test_nemo_asr_model(model,
                   message: str,
                   batch_size: int,
                   manifests: list,
                   probs: bool = False,
                   ):
    """
    transcribing speech to text and then calculate WER and CER metrics

    Parameters:
    -----------

        model:
            model for transribing speech

        batch_size (int):
            batch for model
        
        manifests (List):
            list with path to manifests

        probs (bool) = False:
            if return probabilities of symbols

    Return:
    -------
        test_text (list):
            target text
        
        described_text (list):
            predicted text
        
        probs (list):
            probabilities for symbols after softmax
    """

    test_text = []
    test_path = []
    for path in manifests:

        manif = read_manifest(path)

        for sample in manif:
            if sample["text"] != '':
                test_text.append(sample["text"])
                test_path.append(sample['audio_filepath'])

    if probs:
        logits = model.transcribe(paths2audio_files = test_path,
                                  batch_size = batch_size,
                                  num_workers=2,
                                  logprobs=True)
        probs = []
        for sample in logits:
            probs.append(softmax(sample))
        probs = np.array(probs, 'dtype=object')
        described_text = None

    else:
        described_text = model.transcribe(paths2audio_files=test_path,
                                          batch_size=batch_size,
                                          num_workers=2)
        probs = None

    try:
        word_error = wer(test_text, described_text)
        character_error = cer(test_text, described_text)
        
        print(f'{message}:')
        print('WER:', word_error)
        print('CER:', character_error, '\n')

        return test_text, described_text, probs
    
    except: # some error with calculating metrics because of no prediction

        print('Cannot calculate WER and CER')
        return test_text, described_text, probs