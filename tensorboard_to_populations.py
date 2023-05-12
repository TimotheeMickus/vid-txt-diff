import argparse
import itertools
import pathlib

import pandas as pd
import scipy.stats
import tbparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('output', type=str)
parser.add_argument('tasks', nargs='+', choices=['captioning', 'paraphrase', 'translation', 'paraphrase-captioning', 'paraphrase-translation', 'paraphrase-captioning-translation'])
args = parser.parse_args()
tasks = sorted(set(args.tasks))

logs = pathlib.Path('logs')
dirs = [logs / task for task in tasks]

def dir_to_data(basedir):
    to_concatenate = []
    for name in tqdm.tqdm([str(p) for p in basedir.glob('**/events.*') if not 'multitask' in str(p)], desc=str(basedir)):
        df = tbparse.SummaryReader(name).scalars
        tag, *bummers = [tag for tag in df.tag.unique() if tag.startswith('val/acc-')]
        assert not bummers
        df = df[(df.tag == tag) & (df.step.apply({5, 10, 20, 25}.__contains__))].reset_index()
        df['name'] = name
        to_concatenate.append(df)
    df = pd.concat(to_concatenate)[['name', 'step', 'value']]
    return list(df.itertuples())

pivot, *groups = list(map(dir_to_data, dirs))

aligned = []
while len(pivot) > 0:
    alignments = []
    for M_i in pivot:
        X_is = [min(grp, key=lambda X_j: abs(X_j.value - M_i.value)) for grp in groups]
        alignments.append([M_i, *X_is])
    selected = min(alignments, key=lambda alg: sum(abs(alg[0].value - alg[i + 1].value) for i in range(len(groups))))
    M_i, *X_is = selected
    del pivot[pivot.index(M_i)]
    for X_i, grp in zip(X_is, groups):
        del grp[grp.index(X_i)]
    aligned.append(selected)

print(len(aligned))

final_idx = -1
for idx in range(10, len(aligned) // 2 + 2):
    all_groups = zip(*aligned[:idx])
    all_groups = [[x.value for x in G] for G in all_groups]
    anova = scipy.stats.kruskal(*all_groups)
    print(idx, anova)
    final_idx = idx - 1
    if anova.pvalue < 0.5:  # being very conservative
        break

print(final_idx)

selected = sum(map(list, zip(*aligned[:final_idx])), [])
selected = pd.DataFrame.from_records(selected, columns=['_', 'logfile', 'checkpoint', 'acc'])[['logfile', 'checkpoint', 'acc']]
def to_checkpoint_file(row):
    p = pathlib.Path('model' + row['logfile'][len('log'):]).parent
    tasks = p.parents[1].stem
    comp = 'none'
    if "-" in tasks:
        comp = 'multimodal'
        tasks = '-'.join(sorted(tasks.split('-')))
    return p  / f'{tasks}_{comp}_e{row["checkpoint"]}.pt'

selected['model'] = selected.apply(to_checkpoint_file, axis=1)

with open(args.output, 'w') as ostr:
    for model in selected['model'].to_list():
        print(model, file=ostr)
