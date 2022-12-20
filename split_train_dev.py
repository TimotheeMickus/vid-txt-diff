import argparse
import json
import pathlib
import random

def open_json(fname):
    with open(fname) as fh:
        return json.load(fh)

parser = argparse.ArgumentParser()
parser.add_argument('infile', type=open_json)
parser.add_argument('outdev', type=pathlib.Path)
parser.add_argument('outtrain', type=pathlib.Path)
parser.add_argument('--dev-size', type=int, default=1_000)
args = parser.parse_args()

dev_indices = set(random.sample(range(len(args.infile)), k=args.dev_size))
outdev_items, outtrain_items = [], []
for idx, item in enumerate(args.infile):
    if idx in dev_indices:
        outdev_items.append(item)
    else:
        outtrain_items.append(item)

print('sizes:', len(args.infile), '=>', len(outdev_items), '/', len(outtrain_items))

with open(args.outtrain, 'w') as fh:
    json.dump(outtrain_items, fh)


with open(args.outdev, 'w') as fh:
    json.dump(outdev_items, fh)
