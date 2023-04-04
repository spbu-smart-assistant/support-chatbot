"""
functions for testing models
"""

import re

import numpy as np

from jiwer import wer, cer

# from dataset_utils import read_manifest

def softmax(logits):
    """
    perfom softmax over logits from ASR model
    """

    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])

def test_asr_model(model,
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