"""
functions for datasets
"""

import os
import json
import librosa
import soundfile as sf
from tqdm.auto import tqdm
from typing import List
from pathlib import Path
import re
from collections import defaultdict

#TODO: check if it is correct
def read_manifest(path: str) -> list:
    """
    reads manifest in json format and writes it in list
    """
    manifest = []
    with open(path, 'r') as json_file:
        for line in tqdm(json_file, desc='Reading manifest data'):
            line = line.replace("\n", "")
            data = json.loads(line)
            manifest.append(data)
    return manifest

def commonvoice_to_wav(audio_path: str):
  """
  transforms clips from mp3 format to wav format

  Parameters:
  -----------
      audio_path (str):
          local path to folder with clips

  """
  audio_path_wav = '/'.join(audio_path.split('/')[:-1]) + '/wav_clips'
  if not os.path.exists(audio_path_wav):
    os.mkdir(audio_path_wav)
  for audio_file in os.listdir(audio_path):
    audio, sr = librosa.load(path=audio_path + '/' + audio_file, sr=16000)
    sf.write(file=f'{audio_path_wav}/{audio_file[:-3]}wav', data=audio, 
             samplerate=sr, format='wav')

def get_info_from_tsv(tsv_path: str, wav_audio_folder: str) -> List[tuple]:
  """
  gets information, such as: audio filepath, duration, text - from tsv-file
  and writes it in list audio_info

  Parameters:
  -----------
      tsv_path (str):
          path to tsv-file
      wav_audio_folder (str):
          path where to save wav-clips

  Return:
  -------
      audio_info (List[tuple]):
          list of (audio filepath, duration, text) for each clip 
  """  
  audio_info = []
  is_first_line = True
  with open(tsv_path, 'r') as f:
    for line in tqdm(f):
      if is_first_line:
        is_first_line = False
        continue
      info_list = line.split('\t')
      audio_filepath = f'{wav_audio_folder}/{info_list[1][:-3]}wav'
      duration = librosa.core.get_duration(filename=audio_filepath, sr=16000)
      audio_info.append((audio_filepath, duration, info_list[2]))
  return audio_info

def create_manifest(data: List[tuple], output_name: str, manifest_path: str):
  """
  creates json manifest file from the information stored in parameter 'data'

  Parameters:
  -----------
      data (List[tuple]):
          list of (audio filepath, duration, text) for each clip 
      output_name (str):
          name of manifest file
      manifest_path (str):
          folder for all manifests
  """
  output_path = Path(manifest_path)  / output_name
  output_path.parent.mkdir(exist_ok=True, parents=True)
  with output_path.open(mode='w') as f:
    for wav_path, duration, text in tqdm(data, total=len(data)):
      if wav_path != '':
        f.write(
            json.dumps({'audio_filepath': os.path.abspath(wav_path), 
                        'duration': duration, 
                        'text': text})
            + '\n'
        )

def load_commonvoice_vocab(manifest_root: str):
    """
    Creates set of all words from commonvoice

    Parameters:
    -----------
        manifest_root (str):
            root where all manifest are

    Return:
    -------
        vocab (set):
            set of all unique words in commonvoice data
        
    """
    manifest_dict = []

    for path in os.listdir(manifest_root):
        if path != '.ipynb_checkpoints':
            manifest_data = read_manifest(os.path.join(manifest_root, path))
            manifest_texts = [elem['text'] for elem in manifest_data]

        for text in manifest_texts:
            words = re.findall("\w+", text)
            words = list(filter(lambda x: len(x.strip()) > 3, words))
            manifest_dict.extend(words)

    return set(manifest_dict)

def load_opencorp_vocab(russian_dict_file_path: str):
    """
    Creates set of all words from opencorp file

    Parameters:
    -----------
        russian_dict_file_path (str):
            path to file

    Return:
    -------
        vocab (set):
            set of all unique words in commonvoice data
    """
    russian_dict = []

    with open(russian_dict_file_path, 'r', encoding='utf-8') as dict_file :
        for line in dict_file :
            word = line.split('\t')[0].replace('\n', '').lower()
            if re.findall(r'[0-9]+', word) :
                continue
            if word and not word.isdigit() :
                russian_dict.append(word)

    return set(russian_dict)

def write_processed_manifest(data: List[tuple], original_path: str) -> str:
    """
    creates manifest file using processed data

    Parameters:
    -----------
        data (List[tuple]):
            processed data
        original_path (str):
            path to manifest with unprocessed data

    Return:
    -------
        filepath (str):
            path to manifest with processed data
    """

    original_manifest_name = os.path.basename(original_path)
    new_manifest_name = original_manifest_name.replace(".json", 
                                                       "_processed.json")

    manifest_dir = os.path.split(original_path)[0]
    filepath = os.path.join(manifest_dir, new_manifest_name)
    with open(filepath, 'w') as f:
        for datum in tqdm(data, desc="Writing manifest data"):
            datum = json.dumps(datum)
            f.write(f"{datum}\n")
    print(f"Finished writing manifest: {filepath}")
    return filepath


# calculate the character set
def get_charset(manifest_data: List[tuple]):
    """
    creates set of all unique chars in corpus of texts

    Parameters:
    -----------
        manifest_data (List[tuple]):
            list of (audio filepath, duration, text)

    Return:
    -------
        charset (defaultdict(int)):
            the set of all unique chars
    """

    charset = defaultdict(int)
    for row in tqdm(manifest_data, desc="Computing character set"):
        text = row['text']
        for character in text:
            charset[character] += 1
    return charset

# Preprocessing steps
def lower_text(data: List[tuple]) -> List[tuple]:
    """
    removes capitals in texts

    Parameters:
    -----------
        data (List[tuple]):
            list of (audio filepath, duration, text)

    Return:
    -------
        data (List[tuple]):
            list of (audio filepath, duration, text) without capitals in texts
    """

    data["text"] = data["text"].lower()
    return data

def remove_special_characters(data: List[tuple]) -> List[tuple]:
    """
    removes special characters in texts

    Parameters:
    -----------
        data (List[tuple]):
            list of (audio filepath, duration, text)

    Return:
    -------
        data (List[tuple]):
            list of (audio filepath, duration, text) without special characters
    """

    chars_to_ignore_regex = "[\.\,\?\:\-!;()«»…\]\[/\*–‽+&_\\½√>€™$•¼}{~—=“\"”″‟„]"
    apostrophes_regex = "[’'‘`ʽ']"
    # replace punctuation by space
    data["text"] = re.sub(chars_to_ignore_regex, " ", data["text"]) 
    # replace different apostrophes by one
    data["text"] = re.sub(apostrophes_regex, "'", data["text"]) 
    # merge multiple apostrophes 
    data["text"] = re.sub(r"'+", "'", data["text"])  

    # remove spaces where apostrophe marks a deleted vowel
    # this rule is taken from https://huggingface.co/lucio/wav2vec2-large-xlsr-kinyarwanda-apostrophied
    data["text"] = re.sub(r"([b-df-hj-np-tv-z])' ([aeiou])", r"\1'\2", data["text"])
    # delete apostrophes at the beginning of word
    data["text"] = re.sub(r" '", " ", data["text"])
    # delete apostrophes at the end of word 
    data["text"] = re.sub(r"' ", " ", data["text"])  
    # merge multiple spaces
    data["text"] = re.sub(r" +", " ", data["text"])  
    return data


def replace_diacritics(data: List[tuple]) -> List[tuple]:
    """
    removes diacritics in texts

    Parameters:
    -----------
        data (List[tuple]):
            list of (audio filepath, duration, text)

    Return:
    -------
        data (List[tuple]):
            list of (audio filepath, duration, text) without diacritics
    """
    data["text"] = re.sub(r"[éèëēê]", "e", data["text"])
    data["text"] = re.sub(r"[ãâāá]", "a", data["text"])
    data["text"] = re.sub(r"[úūü]", "u", data["text"])
    data["text"] = re.sub(r"[ôōó]", "o", data["text"])
    data["text"] = re.sub(r"[ćç]", "c", data["text"])
    data["text"] = re.sub(r"[ïī]", "i", data["text"])
    data["text"] = re.sub(r"[ñ]", "n", data["text"])
    return data

#TODO: check if it is correct
def remove_oov_characters(data: List[tuple]) -> List[tuple]:
    """
    removes out-of-vocabulary characters in texts

    Parameters:
    -----------
        data (List[tuple]):
            list of (audio filepath, duration, text)

    Return:
    -------
        data (List[tuple]):
            list of (audio filepath, duration, text) without oov characters
    """
    # oov_regex = "[^ 'aiuenrbomkygwthszdcjfvplxq]"
    oov_regex = "[^ а-я]"
    # delete oov characters
    data["text"] = re.sub(oov_regex, "", data["text"])  
    data["text"] = data["text"].strip()
    return data


# Processing pipeline
def apply_preprocessors(data: List[tuple], preprocessors: list) -> List[tuple]:
    """
    applies preprocessing functions to text

    Parameters:
    -----------
        data (List[tuple]):
            list of (audio filepath, duration, text)
        preprocessors (list):
            list of functions to be applying

    Return:
    -------
        data (List[tuple]):
            list of (audio filepath, duration, text) after preprocessing
    """

    for processor in preprocessors:
        for idx in tqdm(range(len(data)), 
                        desc=f"Applying {processor.__name__}"):
            data[idx] = processor(data[idx])

    print("Finished processing manifest!")
    return data