"""
functions for datasets
"""

import os
import re
import json

#TODO: check if it is correct
def read_manifest(path: str) -> list:
    """
    reads manifest in json format and writes it in list
    """
    manifest = []
    with open(path, 'r') as json_file:
        for line in json_file:
            line = line.replace("\n", "")
            data = json.loads(line)
            manifest.append(data)
    return manifest

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
