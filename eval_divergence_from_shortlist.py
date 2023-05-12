import itertools
import pathlib

import pandas as pd
import torch
import torch.nn.functional as F
import tqdm

from model import Decoder, get_square_mask
import data


shortlist = 'relevant-multimodal-models.txt'
out_fname = 'all-kldivs.multimodal-shortlist.csv'

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
        raise RuntimeError
    else:
        return F.log_softmax(decoder(
            [batch['src_feats'][t] for t in decoder.tasks],
            [batch['src_length'][t] for t in decoder.tasks],
            decoder.tasks,
            batch['tgt'],
        )[:-1], dim=-1).flatten(0, 1)



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

with open(shortlist) as istr:
    models_path = list(map(str.strip, istr))


if not pathlib.Path(out_fname).is_file():
    pd.DataFrame(columns=['model_1', 'model_2', 'kl_div', 'agreement']).to_csv(out_fname, index=False)
all_records = pd.read_csv(out_fname)

def is_valid_model2(candidate, given_model_path):
    return True

full_pbar = tqdm.trange(len(models_path) * (len(models_path) - 1) // 2)

for idx, model1_path in tqdm.tqdm(enumerate(models_path), total=len(models_path), leave=False, position=1):
    model1 = load_model(model1_path)
    models_path2 = models_path
    for model2_path in tqdm.tqdm(models_path[idx + 1:], leave=False, position=2):
        model2 = load_model(model2_path)
        if len(all_records[((all_records.model_1 == model1_path) & (all_records.model_2 == model2_path)) | ((all_records.model_1 == model2_path) & (all_records.model_2 == model1_path))]) >= 2:
            continue

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
            all_kldivs_a, all_kldivs_b, agreement, total = 0., 0., 0., 0.
            
            for batch in test_dataloader:
                p1 = get_all_preds(model1, batch)
                p2 = get_all_preds(model2, batch)
                ids = batch['tgt'].input_ids[...,:-1]
                pad, eos = test_dataset._en_tokenizer.pad_token_id, test_dataset._en_tokenizer.eos_token_id
                true_toks = ((ids != pad) & (ids != eos)).t().contiguous().view(-1, 1)
                all_kldivs_a += (F.kl_div(p1, p2, log_target=True, reduction='none') * true_toks).sum()
                all_kldivs_b += (F.kl_div(p2, p1, log_target=True, reduction='none') * true_toks).sum()
                agreement += ((torch.argmax(p1, dim=-1) == torch.argmax(p2, dim=-1)) * true_toks.squeeze(-1)).sum()
                total += true_toks.sum()
                pbar.update(batch['size'])
            agreement = (agreement / total).item()
            if len(all_records[(all_records.model_1 == model1_path) & (all_records.model_2 == model2_path)]) == 0:
                all_records.loc[len(all_records.index)] = [model1_path, model2_path, all_kldivs_a.item() / len(test_dataset), agreement]
            if len(all_records[(all_records.model_1 == model2_path) & (all_records.model_2 == model1_path)]) == 0:
                all_records.loc[len(all_records.index)] = [model2_path, model1_path, all_kldivs_b.item() / len(test_dataset), agreement]
            all_records.to_csv(out_fname, index=False)
            full_pbar.update()


