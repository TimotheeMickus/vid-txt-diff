import itertools
import pathlib

import pandas as pd
import scipy.stats
import tbparse
import tqdm

logs = pathlib.Path('logs')
C_dir = logs / 'captioning'
P_dir = logs / 'paraphrase'
T_dir = logs / 'translation'

def dir_to_data(basedir, tag):
    to_concatenate = []
    for name in tqdm.tqdm(list(map(str, basedir.glob('**/events.*'))), desc=tag):
        df = tbparse.SummaryReader(name).scalars
        df = df[(df.tag == tag) & (df.step.apply({5, 10, 20, 25}.__contains__))].reset_index()
        df['name'] = name
        to_concatenate.append(df)
    df = pd.concat(to_concatenate)[['name', 'step', 'value']]
    return list(df.itertuples())

Cs = dir_to_data(C_dir, 'val/acc-C')
Ps = dir_to_data(P_dir, 'val/acc-P')
Ts = dir_to_data(T_dir, 'val/acc-T')

aligned = []
while len(Cs) > 0:
    alignments = []
    for C_i in Cs:
        P_i = min(Ps, key=lambda P_j: abs(P_j.value - C_i.value))
        T_i = min(Ts, key=lambda T_j: abs(T_j.value - C_i.value))
        alignments.append([C_i, P_i, T_i])
    selected = min(alignments, key=lambda tri: abs(tri[0].value - tri[1].value) + abs(tri[0].value - tri[2].value))
    C_i, P_i, T_i = selected
    del Cs[Cs.index(C_i)]
    del Ps[Ps.index(P_i)]
    del Ts[Ts.index(T_i)]
    aligned.append(selected)

final_idx = -1
for idx in range(10, len(aligned) + 1):
    A, B, C = zip(*aligned[:idx])
    A, B, C = [x.value for x in A], [x.value for x in B], [x.value for x in C]
    anova = scipy.stats.kruskal(A, B, C)
    print(idx, anova)
    if anova.pvalue < 0.5:  # being very conservative
        final_idx = idx - 1
        break

A, B, C = zip(*aligned[:final_idx])
selected = pd.DataFrame.from_records(A + B + C, columns=['_', 'logfile', 'checkpoint', 'acc'])[['logfile', 'checkpoint', 'acc']]
def to_checkpoint_file(row):
    p = pathlib.Path('model' + row['logfile'][len('log'):]).parent
    return p  / f'{p.parents[1].stem}_none_e{row["checkpoint"]}.pt'

selected['model'] = selected.apply(to_checkpoint_file, axis=1)

with open('relevant-models.txt', 'w') as ostr:
    for model in selected['model'].to_list():
        print(model, file=ostr)
