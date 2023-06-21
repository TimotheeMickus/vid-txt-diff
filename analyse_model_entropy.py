import pandas as pd
import numpy as np
from scipy.stats import kruskal, mannwhitneyu

setups = ['monotask', 'multitask', 'multimodal']

for setup in setups:
    df = pd.read_csv("model_entropy_" + setup + ".csv")
    print("-"*15, "Predictive entropy of", setup.upper(), "models", "-"*15)
    if setup == 'monotask':
        df["task"].replace(to_replace="paraphrase", value="P", inplace=True)
        df["task"].replace(to_replace="captioning", value="C", inplace=True)
        df["task"].replace(to_replace="translation", value="T", inplace=True)

        print(df.task.value_counts())
        P_ent = df[df.task == 'P'].entropy.tolist()
        C_ent = df[df.task == 'C'].entropy.tolist()
        T_ent = df[df.task == 'T'].entropy.tolist()

        print("P_ent mean:", np.array(P_ent).mean())
        print("C_ent mean:", np.array(C_ent).mean())
        print("T_ent mean:", np.array(T_ent).mean())

        print("-"*5, "Kruskall-Wallis test", "-"*5)
        print("all:", kruskal(P_ent, C_ent, T_ent))
        print("P vs C:", kruskal(P_ent, C_ent))
        print("P vs T:", kruskal(P_ent, T_ent))
        print("C vs T:", kruskal(C_ent, T_ent))

        print("-"*5, "Mann-WhitneyU test", "-"*5)
        print("P vs C:", mannwhitneyu(P_ent, C_ent, alternative='two-sided'))
        print("P vs T:", mannwhitneyu(P_ent, T_ent, alternative='two-sided'))
        print("C vs T:", mannwhitneyu(C_ent, T_ent, alternative='two-sided'))

        print("Effect sizes:")
        print("P vs C:", mannwhitneyu(P_ent, C_ent, alternative='two-sided')[0] / (len(P_ent) * len(C_ent)))
        print("P vs T:", mannwhitneyu(P_ent, T_ent, alternative='two-sided')[0] / (len(P_ent) * len(T_ent)))
        print("C vs T:", mannwhitneyu(C_ent, T_ent, alternative='two-sided')[0] / (len(C_ent) * len(T_ent)))

    else:
        df["task"].replace(to_replace="paraphrase", value="P", inplace=True)
        df["task"].replace(to_replace="paraphrase-captioning", value="PC", inplace=True)
        df["task"].replace(to_replace="paraphrase-captioning-translation", value="PCT", inplace=True)
        df["task"].replace(to_replace="paraphrase-translation", value="PT", inplace=True)
        print(df.task.value_counts())

        P_ent = df[df.task == 'P'].entropy.to_list()
        PC_ent = df[df.task == 'PC'].entropy.to_list()
        PCT_ent = df[df.task == 'PCT'].entropy.to_list()
        PT_ent = df[df.task == 'PT'].entropy.to_list()

        print("P_ent mean:", np.array(P_ent).mean())
        print("PC_ent mean:", np.array(PC_ent).mean())
        print("PCT_ent mean:", np.array(PCT_ent).mean())
        print("PT_ent mean:", np.array(PT_ent).mean())

        print("-" * 5, "Kruskall-Wallis test", "-" * 5)
        print("all:", kruskal(P_ent, PC_ent, PCT_ent, PT_ent))
        print("P vs PC:", kruskal(P_ent, PC_ent))
        print("P vs PCT:", kruskal(P_ent, PCT_ent))
        print("P vs PT:", kruskal(P_ent, PT_ent))
        print("PC vs PCT:", kruskal(PC_ent, PCT_ent))
        print("PC vs PT:", kruskal(PC_ent, PT_ent))
        print("PCT vs PT:", kruskal(PCT_ent, PT_ent))

        print("-" * 5, "Mann-WhitneyU test", "-" * 5)
        print("P vs PC:", mannwhitneyu(P_ent, PC_ent, alternative='two-sided'))
        print("P vs PCT:", mannwhitneyu(P_ent, PCT_ent, alternative='two-sided'))
        print("P vs PT:", mannwhitneyu(P_ent, PT_ent, alternative='two-sided'))
        print("PC vs PCT:", mannwhitneyu(PC_ent, PCT_ent, alternative='two-sided'))
        print("PC vs PT:", mannwhitneyu(PC_ent, PT_ent, alternative='two-sided'))
        print("PCT vs PT:", mannwhitneyu(PCT_ent, PT_ent, alternative='two-sided'))