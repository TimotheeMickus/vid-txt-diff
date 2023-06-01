import pandas as pd
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn import metrics
from sklearn.metrics.cluster import homogeneity_score
from scipy.stats import kruskal, mannwhitneyu
from collections import Counter
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, append=True)

tasks = ['P', 'C', 'T']

data_dir = "/users/zosaelai/uncertainty_data/vid-txt-diff"
df = pd.read_csv(os.path.join(data_dir, "concreteness_norms_input_embs_monotask.csv"))

df["task"] = [m.split("/")[0] for m in df.model.tolist()]
df["task"].replace(to_replace="paraphrase", value="P", inplace=True)
df["task"].replace(to_replace="captioning", value="C", inplace=True)
df["task"].replace(to_replace="translation", value="T", inplace=True)
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
            'sil': []
            }

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
C_sil = task_sil[task_sil.task == 'C'].sil.to_list()
T_sil = task_sil[task_sil.task == 'T'].sil.to_list()



# Affinity propagation clustering
def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def compute_purity_balance_data_aff_prop(data, shuffle_data=True):
    word_type_counts = data.word_type.value_counts()
    abstract = data[data.word_type == 'abstract']
    if shuffle_data is True:
        concrete = data[data.word_type == 'concrete'].sample(frac=1)
        concrete = concrete[:word_type_counts['abstract']]
    else:
        concrete = data[data.word_type == 'concrete'][:word_type_counts['abstract']]
    balanced_data = pd.concat([abstract, concrete])
    emb_matrix = np.array(balanced_data.iloc[:, :512])
    true_labels = balanced_data.word_type.tolist()
    aff_prop = AffinityPropagation(damping=0.9).fit(emb_matrix)
    y_pred = aff_prop.labels_
    num_clusters = len(Counter(y_pred))
    num_data = emb_matrix.shape[0]
    purity = purity_score(true_labels, y_pred)
    inv_purity = purity_score(y_pred, true_labels)
    return purity, inv_purity, num_clusters, num_data

print("-"*15, "Affinity propagation: purity scores", "-"*15)

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
C_purity = task_purity_aff_prop[task_purity_aff_prop.task == 'C'].purity.to_list()
T_purity = task_purity_aff_prop[task_purity_aff_prop.task == 'T'].purity.to_list()

P_inv_purity = task_purity_aff_prop[task_purity_aff_prop.task == 'P'].inv_purity.to_list()
C_inv_purity = task_purity_aff_prop[task_purity_aff_prop.task == 'C'].inv_purity.to_list()
T_inv_purity = task_purity_aff_prop[task_purity_aff_prop.task == 'T'].inv_purity.to_list()

print("-"*15, "Silhouette scores", "-"*15)

print("-"*5, "Kruskall-Wallis test", "-"*5)
print("all:", kruskal(P_sil, C_sil, T_sil))
print("P vs C:", kruskal(P_sil, C_sil))
print("P vs T:", kruskal(P_sil, T_sil))
print("C vs T:", kruskal(C_sil, T_sil))

print("-"*5, "Mann-Whitney test", "-"*5)
print("P vs C:", mannwhitneyu(P_sil, C_sil, alternative='two-sided'))
print("P vs T:", mannwhitneyu(P_sil, T_sil, alternative='two-sided'))
print("C vs T:", mannwhitneyu(C_sil, T_sil, alternative='two-sided'))

print("-"*15, "Affinity propagation: purity scores", "-"*15)

print("-"*5, "Kruskall-Wallis test", "-"*5)
print("Purity score")
print("all:", kruskal(P_purity, C_purity, T_purity))
print("P vs C:", kruskal(P_purity, C_purity))
print("P vs T:", kruskal(P_purity, T_purity))
print("C vs T:", kruskal(C_purity, T_purity))

print("Inverse purity score")
print("all:", kruskal(P_inv_purity, C_inv_purity, T_inv_purity))
print("P vs C:", kruskal(P_inv_purity, C_inv_purity))
print("P vs T:", kruskal(P_inv_purity, T_inv_purity))
print("C vs T:", kruskal(C_inv_purity, T_inv_purity))

print("-"*5, "Mann-Whitney test", "-"*5)
print("Purity score")
print("P vs C:", mannwhitneyu(P_purity, C_purity, alternative='two-sided'))
print("P vs T:", mannwhitneyu(P_purity, T_purity, alternative='two-sided'))
print("C vs T:", mannwhitneyu(C_purity, T_purity, alternative='two-sided'))
print("Inverse purity score")
print("P vs C:", mannwhitneyu(P_inv_purity, C_inv_purity, alternative='two-sided'))
print("P vs T:", mannwhitneyu(P_inv_purity, T_inv_purity, alternative='two-sided'))
print("C vs T:", mannwhitneyu(C_inv_purity, T_inv_purity, alternative='two-sided'))