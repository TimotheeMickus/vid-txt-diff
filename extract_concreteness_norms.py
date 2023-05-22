import itertools
import pathlib

import pandas as pd
import torch
import torch.nn.functional as F
import tqdm

from model import Decoder, get_square_mask
import data


cn_df = pd.read_csv('Concreteness_ratings_Brysbaert_et_al_BRM.txt', sep='\t')
data_path = 'data/paraphrase-captioning-translation/1/train.downsampled.no-noise.pt'
dataset =  torch.load(data_path, map_location=torch.device('cpu'))

tokenizer = dataset._en_tokenizer

words_of_interest = set(iter(tokenizer.vocab.keys())) & set(cn_df.Word.to_list())
words_in_train_set = set()
for item in tqdm.tqdm(dataset.items, desc='reading'):
	words_in_train_set = words_in_train_set | set(tokenizer.tokenize(item['tgt']))

words_available = words_of_interest & words_in_train_set
restricted = cn_df[cn_df.Word.apply(words_available.__contains__)].reset_index(drop=True)
restricted['emb_idx'] = restricted.Word.apply(tokenizer.vocab.__getitem__)
restricted.to_csv('ConcNorms_restricted.csv', index=False)
