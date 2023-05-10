import itertools
import pathlib

import pandas as pd
import torch
import torch.nn.functional as F
import tqdm

from model import Decoder, get_square_mask
import data

all_tasks = sorted(['translation', 'paraphrase', 'captioning'])

data_dir = pathlib.Path('data')

test_dataset = data.VatexDataset.get(
    all_tasks,
    data_dir / '-'.join(all_tasks) / ('test.pt'),
    data_dir / "vatex_validation_v1.0.json",
    data_dir / 'val',
    downsample=True,
    noise=None,
)

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
        return [
            F.log_softmax(decoder(
                [batch['src_feats'][t]],
                [batch['src_length'][t]],
                [t],
                batch['tgt'],
            )[:-1], dim=-1).flatten(0, 1)
            for t in decoder.tasks
        ]
    else:
        return [
            F.log_softmax(decoder(
                [batch['src_feats'][t] for t in decoder.tasks],
                [batch['src_length'][t] for t in decoder.tasks],
                decoder.tasks,
                batch['tgt'],
            )[:-1], dim=-1).flatten(0, 1)
        ]

if not pathlib.Path('all-kldivs.csv').is_file():
    pd.concat([
        pd.read_csv('all-but-seeds-kldivs.csv'),
        pd.read_csv('only-seeds-kldivs.csv'),
    ]).to_csv('all-kldivs.csv', index=False)
all_records = pd.read_csv('all-kldivs.csv')

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

for model1_path in tqdm.tqdm(models_path1, leave=True):
    model1 = load_model(model1_path)
    models_path2 = [p for p in models_path if is_valid_model2(p, model1_path)]
    for model2_path in tqdm.tqdm(models_path2, leave=False):
        if model1_path == model2_path:
            continue
        model2 = load_model(model2_path)
        test_dataloader = data.get_dataloader(
            test_dataset,
            batch_size=2048,
            shuffle=False,
        )
        with torch.no_grad(), tqdm.trange(
            len(test_dataset),
            desc=f'{shortform(model1_path)}||{shortform(model2_path)}',
            leave=False,
        ) as pbar:
            all_kdivs = []
            for batch in test_dataloader:
                preds1 = get_all_preds(model1, batch)
                preds2 = get_all_preds(model2, batch)
                for p1, p2 in itertools.product(preds1, preds2):
                    all_kdivs.append(
                        F.kl_div(
                            p1,
                            p2,
                            log_target=True,
                            reduction='sum',
                        ).item()
                    )
                pbar.update(batch['size'])
            all_records.loc[len(all_records.index)] = [model1, model2,  sum(all_kdivs) / len(test_dataset)]
            all_records.to_csv('all-kldivs.csv', index=False)
