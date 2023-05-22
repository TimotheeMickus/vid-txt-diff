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
import os

project_dir = "/users/zosaelai/uncertainty_data/vid-txt-diff"
data_dir = pathlib.Path("/users/zosaelai/uncertainty_data/vid-txt-diff/data")

out_fname = os.path.join(project_dir, 'embeddings.multitask-shortlist.csv')
shortlist_fname = os.path.join(project_dir, 'relevant-multimodal-models.txt')

all_tasks = sorted(['translation', 'paraphrase', 'captioning'])
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

df = pd.read_csv(os.path.join(project_dir, 'ConcNorms_restricted_filtered_for_short_words.csv')).sort_values(by='Conc.M')
abstr_emb_indices = torch.tensor(df.iloc[:1000].emb_idx.to_list(), dtype=torch.long, device=device)
conc_emb_indices = torch.tensor(df.iloc[-1000:].emb_idx.to_list(), dtype=torch.long, device=device)

# data_dir = pathlib.Path('data')
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

word_embs = []
model_names = []
word_types = []
word_indices = []
for idx, model_path in tqdm.tqdm(enumerate(models_path), total=len(models_path), leave=True):
    print('-'*20, 'Model', idx + 1, "of", len(models_path), '-'*20)
    model_name = model_path.split("models/")[1]
    print('Model:', model_name)
    model = load_model(os.path.join(project_dir, model_path)).eval()
    abstract_embs = model.token_embeddings(abstr_emb_indices)
    concrete_embs = model.token_embeddings(conc_emb_indices)
    # abstract words
    word_embs.append(abstract_embs)
    model_names.extend([model_name]*abstract_embs.shape[0])
    word_types.extend(['abstract']*abstract_embs.shape[0])
    abstract_indices = list(abstract_indices.cpu().detach().numpy())
    word_indices.extend(abstract_indices)
    # concrete words
    word_embs.append(concrete_embs)
    model_names.extend([model_name]*concrete_embs.shape[0])
    word_types.extend(['concrete']*concrete_embs.shape[0])
    concrete_indices = list(concrete_indices.cpu().detach().numpy())
    word_indices.extend(concrete_indices)

word_embs = torch.cat(word_embs, dim=0).cpu().detach().numpy()
data = pd.DataFrame(word_embs)
data['model'] = model_names
data['word_type'] = word_types
data['word_index'] = word_indices

print(data.head())
print(data.shape)

csv_file = os.path.join(project_dir, "concreteness_norms_input_embs.csv")
data.to_csv(csv_file, index=None)
print("ALL DONE! Saved embeddings to", csv_file, "!")
