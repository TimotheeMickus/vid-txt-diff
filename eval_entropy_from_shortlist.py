import itertools
import pathlib

import pandas as pd
import torch
import torch.nn.functional as F
import tqdm

from model import Decoder, get_square_mask
#from my_model import Decoder
import data
import os
from scipy.stats import entropy
import numpy as np

project_dir = "/users/zosaelai/uncertainty_data/vid-txt-diff"
data_dir = pathlib.Path("/users/zosaelai/uncertainty_data/vid-txt-diff/data")

all_tasks = sorted(['translation', 'paraphrase', 'captioning'])
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
    print('Load model:', path)
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
        return F.softmax(decoder(
            [batch['src_feats']['paraphrase']],
            [batch['src_length']['paraphrase']],
            ['paraphrase'],
            batch['tgt'],
        )[:-1], dim=-1).flatten(0, 1)
    else:
        return F.softmax(decoder(
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

with open(os.path.join(project_dir,'relevant-models.txt')) as istr:
    models_path = list(map(str.strip, istr))


test_dataloader = data.get_dataloader(
        test_dataset,
        batch_size=2048,
        shuffle=False,
    )

full_pbar = tqdm.trange(len(models_path) * (len(models_path) - 1) // 2)

model_pred_entropy = {}
model_pred_entropy['model'] = []
model_pred_entropy['task'] = []
model_pred_entropy['batch'] = []
model_pred_entropy['sample_idx'] = []
model_pred_entropy['entropy'] = []

for idx, model_path in tqdm.tqdm(enumerate(models_path), total=len(models_path), leave=False, position=1):
    print('Model', idx+1, "of", len(models_path))
    model = load_model(os.path.join(project_dir, model_path))
    entropy_model = []
    for batch_idx, batch in enumerate(test_dataloader):
        pred = get_all_preds(model, batch).cpu().detach().numpy()
        # ids = batch['tgt'].input_ids[..., :-1]
        # pad, eos = test_dataset._en_tokenizer.pad_token_id, test_dataset._en_tokenizer.eos_token_id
        # true_toks = ((ids != pad) & (ids != eos)).t().contiguous().view(-1, 1)
        entropy_batch = entropy(pred, axis=1).mean()
        entropy_model.append(entropy_batch)
    entropy_model = np.mean(entropy_model)
    model_name = model_path.split("models/")[1]
    print("model_name:", model_name)
    print("entropy_model:", entropy_model)
    model_pred_entropy['model'].append(model_name)
    model_pred_entropy['task'].append(model_name.split("/")[0])
    model_pred_entropy['entropy'].append(entropy_model)


# write and save attention totals to CSV
model_pred_entropy = pd.DataFrame(model_pred_entropy)
model_pred_entropy.to_csv(os.path.join(project_dir, "model_entropy_monotask.csv"), index=False)
print('Saved attention scores to:', os.path.join(project_dir, "model_entropy_monotask.csv"))

print("ALL DONE!")

# for idx, model1_path in tqdm.tqdm(enumerate(models_path), total=len(models_path), leave=False, position=1):
#     model1 = load_model(os.path.join(project_dir, model1_path))
#     models_path2 = models_path
#     for model2_path in tqdm.tqdm(models_path[idx + 1:], leave=False, position=2):
#         model2 = load_model(os.path.join(project_dir, model2_path))
#         if len(all_records[((all_records.model_1 == model1_path) & (all_records.model_2 == model2_path)) | ((all_records.model_1 == model2_path) & (all_records.model_2 == model1_path))]) >= 2:
#             continue
#
#         test_dataloader = data.get_dataloader(
#             test_dataset,
#             batch_size=2048,
#             shuffle=False,
#         )
#         with torch.no_grad(), tqdm.trange(
#             len(test_dataset),
#             desc=f'{shortform(model1_path)}||{shortform(model2_path)}',
#             leave=False,
#         ) as pbar:
#             all_kldivs_a, all_kldivs_b, agreement, total = 0., 0., 0., 0.
#
#             for batch in test_dataloader:
#                 p1, p1_attn_weights = get_all_preds(model1, batch)
#                 p2, p2_attn_weights = get_all_preds(model2, batch)
#                 ids = batch['tgt'].input_ids[...,:-1]
#                 pad, eos = test_dataset._en_tokenizer.pad_token_id, test_dataset._en_tokenizer.eos_token_id
#                 true_toks = ((ids != pad) & (ids != eos)).t().contiguous().view(-1, 1)
#                 all_kldivs_a += (F.kl_div(p1, p2, log_target=True, reduction='none') * true_toks).sum()
#                 all_kldivs_b += (F.kl_div(p2, p1, log_target=True, reduction='none') * true_toks).sum()
#                 agreement += ((torch.argmax(p1, dim=-1) == torch.argmax(p2, dim=-1)) * true_toks.squeeze(-1)).sum()
#                 total += true_toks.sum()
#                 pbar.update(batch['size'])
#             agreement = (agreement / total).item()
#             if len(all_records[(all_records.model_1 == model1_path) & (all_records.model_2 == model2_path)]) == 0:
#                 all_records.loc[len(all_records.index)] = [model1_path, model2_path, all_kldivs_a.item() / len(test_dataset), agreement]
#             if len(all_records[(all_records.model_1 == model2_path) & (all_records.model_2 == model1_path)]) == 0:
#                 all_records.loc[len(all_records.index)] = [model2_path, model1_path, all_kldivs_b.item() / len(test_dataset), agreement]
#             all_records.to_csv(out_fname, index=False)
#             full_pbar.update()


