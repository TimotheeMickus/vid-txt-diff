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

shortlist_fname = os.path.join(project_dir, 'relevant-models.txt')

all_tasks = sorted(['translation', 'paraphrase', 'captioning'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

df = pd.read_csv(os.path.join(project_dir, 'ConcNorms_restricted_filtered_for_short_words.csv')).sort_values(by='Conc.M')
abstr_emb_indices = torch.tensor(df.iloc[:1000].emb_idx.to_list(), dtype=torch.long, device=device).unsqueeze(-1)
conc_emb_indices = torch.tensor(df.iloc[-1000:].emb_idx.to_list(), dtype=torch.long, device=device).unsqueeze(-1)

#data_dir = pathlib.Path('data')
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


@torch.no_grad()
def get_all_embs(decoder, item):
    batch = test_dataset.collate([item])
    embs = decoder(
        [batch['src_feats'][t] for t in decoder.tasks],
        [batch['src_length'][t] for t in decoder.tasks],
        decoder.tasks,
        batch['tgt'],
        project_on_vocab=False,
    )
    targets = torch.cat(
        [
            batch['tgt'].input_ids[..., 1:],
            torch.full_like(batch['tgt'].input_ids[..., :1], -1),
        ],
        dim=-1,
    )
    conc_mask = (targets == conc_emb_indices).any(dim=0, keepdims=True)
    abstr_mask = (targets == abstr_emb_indices).any(dim=0, keepdims=True)
    conc_embs = embs.masked_select(conc_mask.t().unsqueeze(-1)).view(-1, embs.size(-1))
    abstr_embs = embs.masked_select(abstr_mask.t().unsqueeze(-1)).view(-1, embs.size(-1))
    conc_idx = targets.masked_select(conc_mask)
    abstr_idx = targets.masked_select(abstr_mask)
    return abstr_embs, abstr_idx, conc_embs, conc_idx


word_embs = []
word_indices = []
model_names = []
word_types =[]
for idx, model_path in tqdm.tqdm(enumerate(models_path), total=len(models_path), leave=True):
    print('-'*20, 'Model', idx + 1, "of", len(models_path), '-'*20)
    model_name = model_path.split("models/")[1]
    print('model_name:', model_name)
    print('model_path:', model_path)
    model = load_model(os.path.join(project_dir, model_path)).eval()
    outputs = []
    for item in tqdm.tqdm(test_dataset.items, desc='items', leave=False):
        outputs.append(get_all_embs(model, item))
    abstract_embs, abstract_indices, concrete_embs, concrete_indices = (
        torch.cat(x).cpu()
        for x in zip(*outputs)
    )

    # abstract
    word_embs.append(abstract_embs)
    model_names.extend([model_name] * abstract_embs.shape[0])
    word_types.extend(['abstract'] * abstract_embs.shape[0])
    abstract_indices = list(abstract_indices.cpu().detach().numpy())
    word_indices.extend(abstract_indices)
    # concrete
    word_embs.append(concrete_embs)
    model_names.extend([model_name] * concrete_embs.shape[0])
    word_types.extend(['concrete'] * concrete_embs.shape[0])
    concrete_indices = list(concrete_indices.cpu().detach().numpy())
    word_indices.extend(concrete_indices)

word_embs = torch.cat(word_embs, dim=0).cpu().detach().numpy()
data = pd.DataFrame(word_embs)
data['model'] = model_names
data['word_type'] = word_types
data['word_index'] = word_indices

print(data.head())
print(data.shape)

csv_file = os.path.join(project_dir, "concreteness_norms_last_hidden_state_embs.csv")
data.to_csv(csv_file, index=None)
print("ALL DONE! Saved embeddings to", csv_file, "!")

