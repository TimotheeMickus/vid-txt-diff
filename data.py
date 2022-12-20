import functools
import itertools
import json
import pathlib
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import tqdm
from transformers import AutoTokenizer, AutoModel


@functools.lru_cache()
def get_embed_with_bert_fn(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    @functools.lru_cache()
    def embed_fn(sentence):
        with torch.no_grad():
            inputs = tokenizer(sentence, return_tensors='pt', truncation=True)
            feats = model(**inputs.to(device)).last_hidden_state
            return feats.squeeze(0).detach().cpu()

    return embed_fn


class VatexDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tasks,
        json_path,
        vid_path,
        downsample=False,
        noise=None,
        device=torch.device('cuda')
    ):
        self._en_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.device = device
        self.noise = noise
        self.downsample = downsample
        self.tasks = tasks
        self.items = []
        with open(json_path, 'r') as data_fh:
            for item in tqdm.tqdm(json.load(data_fh), desc=json_path.name):
                self.items.extend(self.process(item))

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

    def collate(self, items):
        batch = {
            'src_feats': {
                task: pad_sequence([
                    it['src'][task]
                    for it in items
                ]).to(self.device)
                for task in self.tasks
            },
            'src_length': {
                task: torch.tensor([
                    it['src'][task].size(0)
                    for it in items
                ]).to(self.device)
                for task in self.tasks
            },
            'tgt': self._en_tokenizer(
                [it['tgt'] for it in items],
                truncation=True,
                padding=True,
                return_tensors='pt',
            ).to(self.device),
            'size': len(items),
        }
        return batch

    def add_noise(self, features):
        if self.noise is not None:
            features = features + (self.noise * torch.randn_like(features))
        return features.detach()

    def process_translation(self, item):
        src_feat_fn = get_embed_with_bert_fn('bert-base-chinese', self.device)
        if self.downsample:
            all_features = [random.choice(item['chCap'])]
        else:
            all_features = item['chCap']
        all_features = map(src_feat_fn, all_features)
        all_features = map(self.add_noise, all_features)
        return list(all_features)

    def process_paraphrase(self, item):
        src_feat_fn = get_embed_with_bert_fn('bert-base-uncased', self.device)
        if self.downsample:
            all_features = [random.choice(item['enCap'])]
        else:
            all_features = item['enCap']
        all_features = map(src_feat_fn, all_features)
        all_features = map(self.add_noise, all_features)
        return list(all_features)

    def process_captioning(self, item):
        vid_path_ = vid_path / (item["videoID"] + '.npy')
        tensor = torch.from_numpy(np.load(vid_path_)).squeeze(0).to(self.device)
        all_features =  [self.add_noise(tensor)]
        if not self.downsample:
            all_features = all_features * 10
        return all_features

    def process(self, item):
        sources = [
            getattr(self, f'process_{task}')(item)
            for task in self.tasks
        ]
        new_items = [
            {
                'src' : dict(zip(self.tasks, feats)),
                'tgt': tgt,
            }
            for feats in itertools.product(*sources)
            for tgt in  item['enCap']
        ]
        return new_items

    @classmethod
    def get(
        cls,
        tasks,
        filename,
        json_path,
        vid_path,
        downsample=False,
        noise=None,
        device=torch.device('cuda'),
    ):
        filename = pathlib.Path(filename)
        if filename.is_file():
            print(f'retrieving dataset from {filename}')
            return torch.load(filename)
        print(f'building dataset and saving to {filename}')
        filename.parent.mkdir(exist_ok=True, parents=True)
        dataset = cls(
            tasks,
            json_path,
            vid_path,
            downsample=downsample,
            noise=noise,
            device=device,
        )
        torch.save(dataset, filename)
        return dataset


def _get_size_fn():
    _en_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    @functools.lru_cache()
    def _len_tokenize(tgt):
        return len(_en_tokenizer(tgt).input_ids)

    def _size_fn(item):
        return _len_tokenize(item['tgt'])

    return _size_fn


class TokenSampler(torch.utils.data.Sampler):
    """Produce batches with up to `batch_size` tokens in each batch"""

    def __init__(
        self,
        dataset,
        batch_size=256,
        size_fn=_get_size_fn(),
        drop_last=False,
        shuffle=True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.size_fn = size_fn
        self._len = None
        self.drop_last = drop_last
        self.shuffle = True

    def __iter__(self):
        indices = range(len(self.dataset))
        if self.shuffle:
            indices = list(indices)
            random.shuffle(indices)
        i = 0
        selected = []
        numel = 0
        for i in indices:
            if numel + self.size_fn(self.dataset[i]) > self.batch_size:
                if selected:
                    yield selected
                selected = []
                numel = 0
            numel += self.size_fn(self.dataset[i])
            selected.append(i)
        if selected and not self.drop_last:
            yield selected

    def __len__(self):
        if self._len is None:
            self._len = min(
                len(self.dataset),
                round(
                    sum(
                        self.size_fn(self.dataset[i])
                        for i in range(len(self.dataset))
                    ) * 1.03 / self.batch_size  # 1.03 for overshoot
                ),
            )
        return self._len


def get_dataloader(dataset, batch_size=256, shuffle=True):
    return torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate,
            batch_sampler=TokenSampler(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
            ),
        )
