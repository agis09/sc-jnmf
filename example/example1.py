from sc_jnmf import sc_JNMF
import numpy as np
import pandas as pd

from sklearn.metrics.cluster import adjusted_rand_score

df1 = pd.read_csv("../test_data/Pollen_RSEMTopHat.csv", index_col=0)
df2 = pd.read_csv("../test_data/Pollen_Salmon.csv", index_col=0)
label = [i.split('_')[0] for i in df1.columns]
df1.columns = label
df2.columns = label

sc_jnmf = sc_JNMF(df1, df2, rank=8,
                  lambda1=df1.shape[0] / df2.shape[0],
                  lambda4=1)

sc_jnmf.gene_selection()
sc_jnmf.log_scale()
sc_jnmf.normalize()
sc_jnmf.factorize()
sc_jnmf.clustering(cluster_num=len(np.unique(label)))

ari = adjusted_rand_score(label, sc_jnmf.cluster)
print("ARI:", ari)
