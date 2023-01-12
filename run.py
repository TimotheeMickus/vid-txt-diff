import argparse
import collections
import itertools
import pathlib

import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import data
import model
import optim


parser = argparse.ArgumentParser()
parser.add_argument(
    'tasks',
    choices=['translation', 'paraphrase', 'captioning'],
    nargs='+',
)
parser.add_argument(
    '--combination-mode',
    choices=['none', 'multimodal', 'multitask'],
    default='none',
)
parser.add_argument('--epochs', default=26, type=int)
parser.add_argument('--save-every', default=5, type=int)
parser.add_argument('--noise', default=None, type=float)
parser.add_argument('--dynamic-noise', action='store_true')
parser.add_argument('--downsample', action='store_true')
parser.add_argument('--train-path', default=None, type=pathlib.Path)
parser.add_argument(
    '--models-dir',
    default=pathlib.Path('models'),
    type=pathlib.Path,
)
parser.add_argument(
    '--data-dir',
    default=pathlib.Path('data'),
    type=pathlib.Path,
)
parser.add_argument(
    '--logs-dir',
    default=pathlib.Path('logs'),
    type=pathlib.Path,
)
args = parser.parse_args()

args.tasks = sorted(args.tasks)

data_dir = args.data_dir

ext = '.pt'
if args.dynamic_noise:
    ext = '.dynamic' + ext
ext = (f'.noise-{args.noise}' if args.noise is not None else '.no-noise') + ext
if args.downsample:
    ext = '.downsampled' + ext
train_path =  (
    args.train_path.with_suffix(ext)
    if (args.train_path is not None)
    else (data_dir / '-'.join(args.tasks) / ('train' + ext))
)
train_dataset = data.VatexDataset.get(
    args.tasks,
    train_path,
    data_dir / "train.json",
    data_dir / 'val',
    downsample=args.downsample,
    noise=args.noise if not args.dynamic_noise else None,
)
train_dataloader = data.get_dataloader(train_dataset, batch_size=1024)
valid_dataset = data.VatexDataset.get(
    args.tasks,
    data_dir / '-'.join(args.tasks) / ('valid' + ext),
    data_dir / "dev.json",
    data_dir / 'val',
    downsample=True,
    noise=args.noise if not args.dynamic_noise else None,
)
valid_dataloader = data.get_dataloader(
    valid_dataset,
    batch_size=2048,
    shuffle=False,
)

vocab_size = train_dataset._en_tokenizer.vocab_size
padidx = train_dataset._en_tokenizer.pad_token_id
max_len = train_dataset._en_tokenizer.model_max_length
d_src_feats = [
    train_dataset.items[0]['src'][task].size(-1)
    for task in args.tasks
]
decoder = model.Decoder(
    vocab_size,
    padidx,
    max_len,
    d_src_feats=d_src_feats,
    tasks=args.tasks,
    combination_mode=args.combination_mode,
    noise=args.noise if args.dynamic_noise else None,
).to('cuda')
criterion = nn.CrossEntropyLoss(ignore_index=padidx, label_smoothing=0.1)
optimizer = optim.AdaFactorFairSeq(decoder.parameters())

train_batches_repeater = (iter(train_dataloader) for _ in itertools.count())
train_batches = itertools.chain.from_iterable(train_batches_repeater)

losses = collections.defaultdict(
    lambda: collections.deque(maxlen=len(train_dataloader) // 100)
)
accs = collections.defaultdict(
    lambda: collections.deque(maxlen=len(train_dataloader) // 100)
)
global_step = 0
summary_writer = SummaryWriter(args.logs_dir)
if args.combination_mode == 'multimodal':
    task_schedule = itertools.repeat(args.tasks)
else:
    task_schedule = ([task] for task in itertools.cycle(args.tasks))

for epoch in tqdm.trange(args.epochs, desc='Epochs'):
    decoder.train()
    with tqdm.trange(1_000, desc=f'Train {epoch}', leave=True) as pbar:
        postfix = {}
        for step in range(1_000):
            batch = next(train_batches)
            tasks = next(task_schedule)
            optimizer.zero_grad()
            preds = decoder(
                [batch['src_feats'][task] for task in tasks],
                [batch['src_length'][task] for task in tasks],
                tasks,
                batch['tgt'],
            )[:-1].flatten(0, 1)
            gold = batch['tgt'].input_ids.t()[1:].reshape(-1)
            loss = criterion(preds, gold)
            loss.backward()
            optimizer.step()
            relevant_toks = (gold != padidx)
            acc = ((gold == preds.argmax(1)) & relevant_toks).float().sum()
            acc = acc / relevant_toks.float().sum()
            logkey = "".join(t[0].upper() for t in tasks)
            losses[logkey].append(loss.item())
            accs[logkey].append(acc.item())
            summary_writer.add_scalar(
                f'train/loss-{logkey}',
                loss.item(),
                global_step,
            )
            summary_writer.add_scalar(
                f'train/acc-{logkey}',
                acc.item(),
                global_step,
            )
            global_step += 1
            postfix.update({
                f'L-{logkey}': f'{sum(losses[logkey]) / len(losses[logkey]):.3f}',
                f'acc-{logkey}': f'{sum(accs[logkey]) / len(accs[logkey]):.3f}',
            })
            pbar.set_postfix(postfix)
            pbar.update()
    with torch.no_grad(), tqdm.trange(
        len(valid_dataset),
        desc=f'Val {epoch}',
        leave=True,
    ) as pbar_val:
        decoder.eval()
        if args.combination_mode == 'multimodal':
            losses_val = []
            accs_val = []
            for batch in valid_dataloader:
                preds = decoder(
                    [batch['src_feats'][t] for t in args.tasks],
                    [batch['src_length'][t] for t in args.tasks],
                    args.tasks,
                    batch['tgt'],
                )[:-1].flatten(0, 1)
                gold = batch['tgt'].input_ids.t()[1:].reshape(-1)
                loss = criterion(preds, gold)
                relevant_toks = (gold != padidx)
                acc = ((gold == preds.argmax(1)) & relevant_toks).float().sum()
                acc = acc / relevant_toks.float().sum()
                losses_val.append(loss.item())
                accs_val.append(acc.item())
                pbar_val.set_postfix({
                    'L': f'{sum(losses_val) / len(losses_val):.3f}',
                    'acc': f'{sum(accs_val) / len(accs_val):.3f}'
                })
                pbar_val.update(batch['size'])
            logkey = "".join(t[0].upper() for t in args.tasks)
            summary_writer.add_scalar(
                f'val/loss-{logkey}',
                sum(losses_val) / len(losses_val),
                epoch,
            )
            summary_writer.add_scalar(
                f'val/acc-{logkey}',
                sum(accs_val) / len(accs_val),
                epoch,
            )
        else:
            losses_val = {task: [] for task in args.tasks}
            accs_val = {task: [] for task in args.tasks}
            for batch in valid_dataloader:
                for task in args.tasks:
                    preds = decoder(
                        [batch['src_feats'][task]],
                        [batch['src_length'][task]],
                        [task],
                        batch['tgt'],
                    )[:-1].flatten(0, 1)
                    gold = batch['tgt'].input_ids.t()[1:].reshape(-1)
                    loss = criterion(preds, gold)
                    relevant_toks = (gold != padidx)
                    acc = ((gold == preds.argmax(1)) & relevant_toks).float().sum()
                    acc = acc / relevant_toks.float().sum()
                    losses_val[task].append(loss.item())
                    accs_val[task].append(acc.item())
                postfix = {}
                for task in args.tasks:
                    postfix.update({
                        f'L-{task[0].upper()}': f'{sum(losses_val[task]) / len(losses_val[task]):.3f}',
                        f'acc-{task[0].upper()}': f'{sum(accs_val[task]) / len(accs_val[task]):.3f}'
                    })
                pbar_val.set_postfix(postfix)
                pbar_val.update(batch['size'])
            for task in args.tasks:
                summary_writer.add_scalar(
                    f'val/loss-{task[0].upper()}',
                    sum(losses_val[task]) / len(losses_val[task]),
                    epoch,
                )
                summary_writer.add_scalar(
                    f'val/acc-{task[0].upper()}',
                    sum(accs_val[task]) / len(accs_val[task]),
                    epoch,
                )
    if epoch % args.save_every == 0:
        fname = f'{"-".join(args.tasks)}_{args.combination_mode}_e{epoch}.pt'
        torch.save(
            decoder.state_dict(),
            args.models_dir / fname,
        )
