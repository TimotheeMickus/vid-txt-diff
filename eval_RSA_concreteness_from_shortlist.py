import itertools
import pathlib

import pandas as pd
import torch
import torch.nn.functional as F
import torchmetrics
import tqdm
import scipy.stats

from model import Decoder, get_square_mask
import data

out_fname = 'all-concRSA.multimodal-shortlist.csv'
shortlist_fname = 'relevant-multimodal-models.txt'

all_tasks = sorted(['translation', 'paraphrase', 'captioning'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('ConcNorms_restricted.csv').sort_values(by='Conc.M')
abstr_emb_indices = torch.tensor(df.iloc[:1000].emb_idx.to_list(), dtype=torch.long)
conc_emb_indices = torch.tensor(df.iloc[-1000:].emb_idx.to_list(), dtype=torch.long)

data_dir = pathlib.Path('data')
sample_dataset_path = data_dir / '-'.join(all_tasks) / '1k-test-sample.txt'

assert sample_dataset_path.is_file()

test_dataset = torch.load(sample_dataset_path, map_location=device)

vocab_size = test_dataset._en_tokenizer.vocab_size
padidx = test_dataset._en_tokenizer.pad_token_id
max_len = test_dataset._en_tokenizer.model_max_length
d_src_feats = [
    test_dataset.items[0]['src'][task].size(-1)
    for task in all_tasks
]

del test_dataset


def find_in_path(path, default, patterns):
    if type(patterns[0]) not in {list, tuple}:
        patterns = ([p, p] for p in patterns)
    for pattern, value in patterns:
        if pattern in path:
            return value
    return default

def load_model(path, device=device):
    mode = find_in_path(
        path,
        'none',
        ['multimodal', 'multitask'],
    )
    noise = find_in_path(
        path,
        None,
        [['-n-0.5', 0.5], ['-n-1.0', 1.0], ['-n-1.5', 1.5]],
    )
    tasks = [t for t in all_tasks if t in path]
    feats = [f for i, f in enumerate(d_src_feats) if all_tasks[i] in path]
    decoder = Decoder(
        vocab_size,
        padidx,
        max_len,
        d_src_feats=feats,
        tasks=tasks,
        combination_mode=mode,
        noise=noise,
    )
    decoder.load_state_dict(torch.load(path, map_location=device))
    decoder.to(device)
    decoder.eval()
    return decoder


with open(shortlist_fname) as istr:
    models_path = list(map(str.strip, istr))


def is_valid_model2(candidate, given_model_path):
    return True


def shortform(model_name, as_str=True):
    comps = model_name.split('/')
    tasks = ''.join(c[0].upper() for c in comps[1].split('-'))
    noise = comps[2].split('-')[-1]
    seed = comps[-2]
    *_, mode, epoch = comps[-1].split('_')
    epoch = epoch[1:-3]
    info = [tasks, noise, mode, seed, epoch]
    if as_str:
        return '/'.join(info)
    return info


if pathlib.Path(out_fname).is_file():
    all_records = pd.read_csv(out_fname)
    shortform_models_path = {shortform(mp) for mp in models_path}
    all_records = all_records[all_records.model1.apply(shortform_models_path.__contains__) & all_records.model2.apply(shortform_models_path.__contains__)].reset_index(drop=True)
    all_records.to_csv(out_fname, index=False)
else:
    all_records = pd.DataFrame(columns=['model1', 'model2', 'rsa_rho_abstr', 'rsa_pval_abstr', 'rsa_rho_conc', 'rsa_pval_conc'])


@torch.no_grad()
def compute_rsa(X1, X2):
    assert X1.size() == X2.size()
    X1, X2, = X1.unsqueeze(0), X2.unsqueeze(0)
    X1 = torch.cdist(X1, X1).squeeze(0)
    X2 = torch.cdist(X2, X2).squeeze(0)
    X1 = X1.masked_select(torch.ones_like(X1).triu(diagonal=1).bool()).cpu().numpy()
    X2 = X2.masked_select(torch.ones_like(X2).triu(diagonal=1).bool()).cpu().numpy()
    return scipy.stats.spearmanr(X1, X2)


def should_skip(model1_path, model2_path):
    return len(
        all_records[
            (all_records.model1 == shortform(model1_path)) &
            (all_records.model2 == shortform(model2_path))
        ]
    ) != 0


with tqdm.trange(len(models_path) * (len(models_path) - 1) // 2) as full_pbar, torch.no_grad():
  for idx, model1_path in tqdm.tqdm(enumerate(models_path), total=len(models_path), leave=True):
    model1 = load_model(model1_path).eval()
    abstr_embs_model1 = model1.token_embeddings(abstr_emb_indices)
    conc_embs_model1 = model1.token_embeddings(conc_emb_indices)
    del model1
    for model2_path in tqdm.tqdm(models_path[idx + 1:], leave=False):
        if should_skip(model1_path, model2_path):
          full_pbar.update()
          continue
        model2 = load_model(model2_path).eval()
        abstr_embs_model2 = model2.token_embeddings(abstr_emb_indices)
        conc_embs_model2 = model2.token_embeddings(conc_emb_indices)
        del model2
        abstr_rsa_scores = compute_rsa(abstr_embs_model1, abstr_embs_model2)
        conc_rsa_scores = compute_rsa(conc_embs_model1, conc_embs_model2)
        all_records.loc[len(all_records.index)] = [shortform(model1_path), shortform(model2_path), *abstr_rsa_scores, *conc_rsa_scores]
        all_records.loc[len(all_records.index)] = [shortform(model2_path), shortform(model1_path), *abstr_rsa_scores, *conc_rsa_scores]
        all_records.to_csv(out_fname, index=False)
        full_pbar.update()

