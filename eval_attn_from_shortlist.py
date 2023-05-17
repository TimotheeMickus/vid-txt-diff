import itertools
import pathlib

import pandas as pd
import torch
import torch.nn.functional as F
import tqdm

# from model import Decoder, get_square_mask
from my_model import Decoder
import data
import os
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import json


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
        raise RuntimeError
    else:
        last_hidden_state, last_hidden_weights = decoder(
            [batch['src_feats'][t] for t in decoder.tasks],
            [batch['src_length'][t] for t in decoder.tasks],
            decoder.tasks,
            batch['tgt'],
        )

        pred = F.log_softmax(last_hidden_state[:-1], dim=-1).flatten(0, 1)
        
        # return F.log_softmax(decoder(
        #     [batch['src_feats'][t] for t in decoder.tasks],
        #     [batch['src_length'][t] for t in decoder.tasks],
        #     decoder.tasks,
        #     batch['tgt'],
        # )[:-1], dim=-1).flatten(0, 1)
        # print('last_hidden_state:', last_hidden_state)
        # print('last_hidden_weights:', last_hidden_weights.shape)
        return last_hidden_state, last_hidden_weights



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

with open(os.path.join(project_dir,'relevant-multimodal-models.txt')) as istr:
    models_path = list(map(str.strip, istr))


out_fname = 'all-kldivs.shortlist.csv'
if not pathlib.Path(out_fname).is_file():
    pd.DataFrame(columns=['model_1', 'model_2', 'kl_div', 'agreement']).to_csv(out_fname, index=False)
all_records = pd.read_csv(out_fname)

def is_valid_model2(candidate, given_model_path):
    return True

def plot_attention_heatmap(attn_matrix, model_name):
    sns_plot = sns.heatmap(attn_matrix.cpu().detach().numpy())
    model_name = model_name.replace("/", "_")
    fig_filename = os.path.join(project_dir, 'attn_heatmaps', model_name + ".png")
    plt.savefig(fig_filename)
    print('Saved attention heatmap to', fig_filename)
    plt.close()

test_dataloader = data.get_dataloader(
        test_dataset,
        batch_size=2048,
        shuffle=False,
    )

full_pbar = tqdm.trange(len(models_path) * (len(models_path) - 1) // 2)

model_task_attn = {t: [] for t in all_tasks}
model_task_attn['model'] = []
model_task_attn['task'] = []
model_task_attn['batch'] = []
model_task_attn['batch_size'] = []
model_task_attn['sample_idx'] = []
for idx, model_path in tqdm.tqdm(enumerate(models_path), total=len(models_path), leave=False, position=1):
    print('Model', idx+1, "of", len(models_path))
    model = load_model(os.path.join(project_dir, model_path))

    model_name = model_path.split("models/")[1]
    for batch_idx, batch in enumerate(test_dataloader):
        print('-'*15, 'batch', batch_idx, '-'*15)
        print('batch size:', batch['size'])
        pred, attn_weights = get_all_preds(model, batch)
        print('attn_weights:', attn_weights.shape)
        # attn_matrix_sum = attn_weights.sum(dim=0)
        # print('attn_matrix_sum:', attn_matrix_sum.shape)
        # if batch_idx == 0:
        #     plot_attention_heatmap(attn_weights[0].T, model_name+"_batch0_example0")
        attn_tokens_sum = attn_weights.sum(dim=1)
        print('attn_tokens_sum:', attn_tokens_sum.shape)
        # for each example, sum up attention weights by task
        for sample_idx in range(attn_tokens_sum.shape[0]):
            print('--- sample', sample_idx, '---')
            boundaries = {t: batch['src_length'][t].cpu().detach().numpy() for t in model.tasks}
            print('tasks:', model.tasks)
            start_index = 0
            task_attn = {t: 0 for t in all_tasks}
            for task in boundaries:
                print('task:', task)
                end_index = start_index + boundaries[task].max()
                print('start_index:', start_index)
                print('end_index:', end_index)
                task_attn_sum = attn_tokens_sum[sample_idx][start_index:end_index].sum().item()
                task_attn[task] = task_attn_sum
                start_index = end_index
            model_task_attn['model'].append(model_name)
            model_task_attn['task'].append(model_name.split("/")[0])
            model_task_attn['batch'].append(batch_idx)
            model_task_attn['batch_size'].append(batch['size'])
            model_task_attn['sample_idx'].append(sample_idx)
            for task in task_attn:
                model_task_attn[task].append(task_attn[task])


# write and save attention totals to CSV
model_task_attn = pd.DataFrame(model_task_attn)
model_task_attn.to_csv(os.path.join(project_dir, "model_attn_by_sample.csv"), index=False)
print('Saved attention scores to:', os.path.join(project_dir, "model_attn_by_sample.csv"))

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


