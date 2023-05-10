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

all_tasks = sorted(['translation', 'paraphrase', 'captioning'])

data_dir = pathlib.Path('data')
sample_dataset_path = data_dir / '-'.join(all_tasks) / '1k-test-sample.txt'

if not sample_dataset_path.is_file():
    import random
    base_test_dataset = data.VatexDataset.get(
        all_tasks,
        data_dir / '-'.join(all_tasks) / ('test.pt'),
        data_dir / "vatex_validation_v1.0.json",
        data_dir / 'val',
        downsample=True,
        noise=None,
    )
    base_test_dataset.items = sorted(
        random.sample(base_test_dataset.items, 1000),
        key=lambda d: sum(f.size(0) for f in d['src'].values()) / len(d['src']),
    )
    torch.save(base_test_dataset, sample_dataset_path)

test_dataset = torch.load(sample_dataset_path)

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

def load_model(path, device=torch.device('cuda')):
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
    decoder.load_state_dict(torch.load(path))
    decoder.to(device)
    decoder.eval()
    return decoder

@torch.no_grad()
def get_all_preds(decoder, batch):
    if decoder.combination_mode == 'multitask':
        raise NotImplementedError
    return decoder(
        [batch['src_feats'][t] for t in decoder.tasks],
        [batch['src_length'][t] for t in decoder.tasks],
        decoder.tasks,
        batch['tgt'],
        project_on_vocab=False,
    )[:-1].flatten(0, 1)

models_path = [
    str(p)
    for p in pathlib.Path('models').glob('**/*.pt')
    if 'multitask' not in str(p) #and '/1/' in str(p)
]
models_path1 = [
    str(p)
    for p in pathlib.Path('models').glob('**/*.pt')
    if 'multitask' not in str(p) and '/1/' in str(p)
]

def is_valid_model2(candidate, given_model_path):
    if len(
        all_records[
            (all_records.model_1 == given_model_path) &
            (all_records.model_2 == candidate)
        ]
    ) != 0:
        return False
    return shortform(candidate, as_str=False)[0] == shortform(given_model_path, as_str=False)[0]



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

def get_batches():
    return (test_dataset.collate([item]) for item in test_dataset.items)


if pathlib.Path('all-RSA.csv').is_file():
    all_records = pd.read_csv('all-RSA.csv')
else:
    all_records = pd.DataFrame(columns=['model1', 'model2', 'rsa_rho', 'rsa_pval'])


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

for idx, model1_path in tqdm.tqdm(enumerate(models_path1), total=len(models_path1), leave=True):
    model1 = load_model(model1_path).eval()
    embs_model1 = []
    for batch in tqdm.tqdm(get_batches(), total=len(test_dataset), desc=shortform(model1_path), leave=False):
        embs_model1.append(get_all_preds(model1, batch))
    embs_model1 = torch.cat(embs_model1)
    for model2_path in tqdm.tqdm(models_path1[idx + 1:], leave=False):
        if should_skip(model1_path, model2_path): continue
        model2 = load_model(model2_path).eval()
        embs_model2 = []
        for batch in tqdm.tqdm(get_batches(), total=len(test_dataset), desc=shortform(model2_path), leave=False):
            embs_model2.append(get_all_preds(model2, batch))
        del model2
        embs_model2 = torch.cat(embs_model2)
        rsa_scores = compute_rsa(embs_model1, embs_model2)
        all_records.loc[len(all_records.index)] = [shortform(model1_path), shortform(model2_path), *rsa_scores]
        all_records.loc[len(all_records.index)] = [shortform(model2_path), shortform(model1_path), *rsa_scores]
        all_records.to_csv('all-RSA.csv', index=False)
