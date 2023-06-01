import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn import metrics
from sklearn.metrics.cluster import homogeneity_score
from scipy.stats import kruskal, mannwhitneyu
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, append=True)

tasks = ['P', 'PC', 'PCT', 'PT']
data_dir = "/users/zosaelai/uncertainty_data/vid-txt-diff"
df = pd.read_csv(os.path.join(data_dir, "concreteness_norms_last_hidden_state_embs_multitask.csv"))

df["task"] = [m.split("/")[0] for m in df.model.tolist()]
df["task"].replace(to_replace="paraphrase", value="P", inplace=True)
df["task"].replace(to_replace="paraphrase-captioning", value="PC", inplace=True)
df["task"].replace(to_replace="paraphrase-captioning-translation", value="PCT", inplace=True)
df["task"].replace(to_replace="paraphrase-translation", value="PT", inplace=True)
print(df.task.value_counts())


def compute_silhouette_balance_data(data, shuffle_data=True):
    word_type_counts = data.word_type.value_counts()
    abstract = data[data.word_type == 'abstract']
    if shuffle_data is True:
        concrete = data[data.word_type == 'concrete'].sample(word_type_counts['abstract'])
    else:
        concrete = data[data.word_type == 'concrete'][:word_type_counts['abstract']]
    balanced_data = pd.concat([abstract, concrete])
    emb_matrix = np.array(balanced_data.iloc[:,:512])
    true_labels = balanced_data.word_type.tolist()
    return metrics.silhouette_score(emb_matrix, true_labels)

print("-"*15, "Silhouette scores", "-"*15)
task_sil = {'task': [],
            'sil': []}

for task in tasks:
    print('Task:', task.upper())
    task_models = df[df.task == task]
    model_sil = task_models.groupby("model").apply(lambda x: compute_silhouette_balance_data(x))
    print('mean:', model_sil.mean())
    print('std:', model_sil.std())
    task_sil['sil'].extend(model_sil.tolist())
    task_sil['task'].extend([task]*len(model_sil))

task_sil = pd.DataFrame.from_dict(task_sil)
P_sil = task_sil[task_sil.task == 'P'].sil.to_list()
PC_sil = task_sil[task_sil.task == 'PC'].sil.to_list()
PCT_sil = task_sil[task_sil.task == 'PCT'].sil.to_list()
PT_sil = task_sil[task_sil.task == 'PT'].sil.to_list()


print("-"*5, "Affinity propagation", "-"*5)
def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

# Affinity propagation clustering
def compute_purity_balance_data_aff_prop(data, shuffle_data=True):
    word_type_counts = data.word_type.value_counts()
    abstract = data[data.word_type == 'abstract']
    if shuffle_data is True:
        concrete = data[data.word_type == 'concrete'].sample(frac=1)
        concrete = concrete[:word_type_counts['abstract']]
    else:
        concrete = data[data.word_type == 'concrete'][:word_type_counts['abstract']]
    balanced_data = pd.concat([abstract, concrete])
    emb_matrix = np.array(balanced_data.iloc[:,:512])
    true_labels = balanced_data.word_type.tolist()
    aff_prop = AffinityPropagation().fit(emb_matrix)
    y_pred = aff_prop.labels_
    num_clusters = len(Counter(y_pred))
    num_data = emb_matrix.shape[0]
    purity = purity_score(true_labels, y_pred)
    inv_purity = purity_score(y_pred, true_labels)
    return purity, inv_purity, num_clusters, num_data


task_purity_aff_prop = {
                        'task': [],
                        'purity': [],
                        'inv_purity': [],
                        'num_clusters': []
                        }
for task in tasks:
    print('Task:', task.upper())
    task_models = df[df.task == task]
    purity_scores = task_models.groupby("model").apply(lambda x: compute_purity_balance_data_aff_prop(x))
    model_purity, inv_model_purity, num_clusters, num_data = zip(*purity_scores)
    model_purity = np.array(model_purity)
    inv_model_purity = np.array(inv_model_purity)
    num_clusters = np.array(num_clusters)
    num_data = np.array(num_data)
    print('purity mean:', model_purity.mean())
    print('purity std:', model_purity.std())
    print("-"*15)
    print('inv purity mean:', inv_model_purity.mean())
    print('inv purity std:', inv_model_purity.std())
    print("-"*15)
    print('num_clusters mean:', num_clusters.mean())
    print('num_clusters std:', num_clusters.std())
    print('num_data:', num_data.mean())
    print("-"*15)
    task_purity_aff_prop['purity'].extend(model_purity.tolist())
    task_purity_aff_prop['inv_purity'].extend(inv_model_purity.tolist())
    task_purity_aff_prop['num_clusters'].extend(num_clusters.tolist())
    task_purity_aff_prop['task'].extend([task]*len(model_purity))


task_purity_aff_prop = pd.DataFrame.from_dict(task_purity_aff_prop)
P_purity = task_purity_aff_prop[task_purity_aff_prop.task == 'P'].purity.to_list()
PC_purity = task_purity_aff_prop[task_purity_aff_prop.task == 'PC'].purity.to_list()
PCT_purity = task_purity_aff_prop[task_purity_aff_prop.task == 'PCT'].purity.to_list()
PT_purity = task_purity_aff_prop[task_purity_aff_prop.task == 'PT'].purity.to_list()


P_inv_purity = task_purity_aff_prop[task_purity_aff_prop.task == 'P'].inv_purity.to_list()
PC_inv_purity = task_purity_aff_prop[task_purity_aff_prop.task == 'PC'].inv_purity.to_list()
PCT_inv_purity = task_purity_aff_prop[task_purity_aff_prop.task == 'PCT'].inv_purity.to_list()
PT_inv_purity = task_purity_aff_prop[task_purity_aff_prop.task == 'PT'].inv_purity.to_list()

print("-"*15, "Silhouette scores", "-"*15)
print("-"*5, "Kruskall-Wallis test", "-"*5,)
print("all:", kruskal(P_sil, PC_sil, PCT_sil, PT_sil))
print("P vs PC:", kruskal(P_sil, PC_sil))
print("P vs PCT:", kruskal(P_sil, PCT_sil))
print("P vs PT:", kruskal(P_sil, PT_sil))
print("PC vs PCT:", kruskal(PC_sil, PCT_sil))
print("PC vs PT:", kruskal(PC_sil, PT_sil))
print("PCT vs PT:", kruskal(PCT_sil, PT_sil))

print("-"*5, "Mann-Whitney test", "-"*5,)
print("P vs PC:", mannwhitneyu(P_sil, PC_sil, alternative='two-sided'))
print("P vs PCT:", mannwhitneyu(P_sil, PCT_sil, alternative='two-sided'))
print("P vs PT:", mannwhitneyu(P_sil, PT_sil, alternative='two-sided'))
print("PC vs PCT:", mannwhitneyu(PC_sil, PCT_sil, alternative='two-sided'))
print("PC vs PT:", mannwhitneyu(PC_sil, PT_sil, alternative='two-sided'))
print("PCT vs PT:", mannwhitneyu(PCT_sil, PT_sil, alternative='two-sided'))

print("-"*15, "Affinity propagation", "-"*15)

print("-"*5, "Kruskall-Wallis test", "-"*5)
print("Purity scores:")
print("all:", kruskal(P_purity, PC_purity, PCT_purity, PT_purity))
print("P vs PC:", kruskal(P_purity, PC_purity))
print("P vs PCT:", kruskal(P_purity, PCT_purity))
print("P vs PT:", kruskal(P_purity, PT_purity))
print("PC vs PCT:", kruskal(PC_purity, PCT_purity))
print("PC vs PT:", kruskal(PC_purity, PT_purity))
print("PCT vs PT:", kruskal(PCT_purity, PT_purity))
print("Inverse purity scores:")
print("all:", kruskal(P_inv_purity, PC_inv_purity, PCT_inv_purity, PT_inv_purity))
print("P vs PC:", kruskal(P_inv_purity, PC_inv_purity))
print("P vs PCT:", kruskal(P_inv_purity, PCT_inv_purity))
print("P vs PT:", kruskal(P_inv_purity, PT_inv_purity))
print("PC vs PCT:", kruskal(PC_inv_purity, PCT_inv_purity))
print("PC vs PT:", kruskal(PC_inv_purity, PT_inv_purity))
print("PCT vs PT:", kruskal(PCT_inv_purity, PT_inv_purity))

print("-"*5, "Mann-Whitney test", "-"*5)
print("Purity score")
print("P vs PC:", mannwhitneyu(P_purity, PC_purity, alternative='two-sided'))
print("P vs PCT:", mannwhitneyu(P_purity, PCT_purity, alternative='two-sided'))
print("P vs PT:", mannwhitneyu(P_purity, PT_purity, alternative='two-sided'))
print("PC vs PCT:", mannwhitneyu(PC_purity, PCT_purity, alternative='two-sided'))
print("PC vs PT:", mannwhitneyu(PC_purity, PT_purity, alternative='two-sided'))
print("PCT vs PT:", mannwhitneyu(PCT_purity, PT_purity, alternative='two-sided'))
print("Inverse purity score")
print("P vs PC:", mannwhitneyu(P_inv_purity, PC_inv_purity, alternative='two-sided'))
print("P vs PCT:", mannwhitneyu(P_inv_purity, PCT_inv_purity, alternative='two-sided'))
print("P vs PT:", mannwhitneyu(P_inv_purity, PT_inv_purity, alternative='two-sided'))
print("PC vs PCT:", mannwhitneyu(PC_inv_purity, PCT_inv_purity, alternative='two-sided'))
print("PC vs PT:", mannwhitneyu(PC_inv_purity, PT_inv_purity, alternative='two-sided'))
print("PCT vs PT:", mannwhitneyu(PCT_inv_purity, PT_inv_purity, alternative='two-sided'))