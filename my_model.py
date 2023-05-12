import torch
import torch.nn as nn
import torch.nn.functional as F
import my_decoder

get_square_mask = nn.Transformer.generate_square_subsequent_mask


def str_to_activation_fn(act, format='function'):
    assert format in {'function', 'layer'}, 'Wrong specifications'
    if format == 'function':
        return getattr(F, act.lower())
    elif format == 'layer':
        return getattr(nn, act)()


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        padding_idx,
        max_len,
        d_src_feats=[512],
        tasks=['paraphrase'],
        combination_mode='none',
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation='ReLU',
        num_layers=6,
        noise=None,
    ):
        assert combination_mode in {'multitask', 'multimodal', 'none'}
        assert len(tasks) == len(d_src_feats)
        if combination_mode == 'none':
            assert len(tasks) == 1
        self.tasks = tasks

        super().__init__()
        self.token_embeddings = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=padding_idx,
        )
        self.max_len = max_len
        self.position_embeddings = nn.Embedding(
            max_len,
            d_model,
        )
        decoder_layer = my_decoder.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=str_to_activation_fn(activation),
        )
        self.encoder_projs = nn.ModuleDict({
            task: nn.Sequential(
                nn.Dropout(),
                nn.Linear(dsf, dim_feedforward),
                str_to_activation_fn(activation, format='layer'),
                nn.Linear(dim_feedforward, d_model),
            )
            for task, dsf in zip(tasks, d_src_feats)
        })
        self.combination_mode = combination_mode
        self.decoder = my_decoder.TransformerDecoder(decoder_layer, num_layers)
        self.vocab_proj = nn.Linear(d_model, vocab_size)
        self.noise = noise
        self._init()

    def _init(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            else:  # gain parameters of the layer norm
                nn.init.ones_(param)

    def forward(self, src_feats, src_lengths, tasks, tgt, project_on_vocab=True):
        tgt_ids = tgt.input_ids.t()
        T, B = tgt_ids.size()

        assert len(src_feats) == len(src_lengths) == len(tasks)

        src_feats_to_combine, src_masks_to_combine = [], []
        for feats, lengths, task in zip(src_feats, src_lengths, tasks):
            S, B_, F = feats.size()
            assert B == B_
            src_indices = torch.arange(S).to(tgt_ids.device).unsqueeze(1)
            src_masks_to_combine.append(src_indices > lengths.unsqueeze(0))
            if self.noise is not None:
                noise = torch.randn_like(feats) * self.noise
                feats = feats + noise
            src_feats_to_combine.append(self.encoder_projs[task](feats))
        src_feats = torch.cat(src_feats_to_combine, dim=0)
        src_padding_mask = torch.cat(src_masks_to_combine, dim=0).t()

        tgt_mask = get_square_mask(T, device=tgt_ids.device)
        tgt_padding_mask = (tgt_ids == self.token_embeddings.padding_idx).t()

        tgt_seq = self.token_embeddings(tgt_ids)
        tgt_seq = tgt_seq + self.position_embeddings.weight[:T].unsqueeze(1)
        last_hidden_state, last_hidden_attn_weights = self.decoder(
            tgt=tgt_seq,
            memory=src_feats,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        if project_on_vocab:
            return self.vocab_proj(last_hidden_state), last_hidden_attn_weights
        else:
            return last_hidden_state, last_hidden_attn_weights
