import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import Normalizer
from ._joint_nmf_gpu import Joint_NMF_GPU
from ._joint_nmf_cpu import Joint_NMF_CPU
from ._matrix_init import random_init


class sc_JNMF:
    """
    An analysis tool using Joint-NMF for single cell gene expression profiles.

    Parameters
    ----------

    D1 : pandas DataFrame
        Gene expression matrix of pandas dataframe (row : gene, col : cell).
        D1 columns must be same as D2 columns.
    D2 : pandas DataFrame
        Gene expression matrix of pandas dataframe (row : gene, col : cell).
        D2 columns must be same as D1 columns.

    rank : int
        The rank in matrix factorization.

    lambda1 : float, default 1.0
        The coefficient (parameter) of |D2-W2*H|_F^2 in the objective function.

    lambda2 : float, default 0.0
        The coefficient (parameter) of |W1| (l1 or l2 norm) in the objective function.

    lambda3 : float, default 0.0
        The coefficient (parameter) of |W2| (l1 or l2 norm) in the objective function.

    lambda4 : float, default 1.0
        The coefficient (parameter) of |H| (l1 or l2 norm) in the objective function.

    W1 : 2d ndarray or None, default None
        Initial value of factorized matrix (gene * rank).

    W2 : 2d ndarray or None, default None
        Initial value of factorized matrix (gene * rank).

    H : 2d ndarray or None, default None
        Initial value of factorized matrix (rank * cell).

    geneset1 : None
        The result of 'gene_selection' in geneset1.

    geneset2 : None
        The result of 'gene_selection' in geneset2.

    cluster : None
        The result of cell clustering.

    """

    def __init__(self, D1, D2, rank, lambda1=1., lambda2=0.,
                 lambda3=0., lambda4=1., W1=None, W2=None, H=None,
                 geneset1=None, geneset2=None, cluster=None):
        self.D1 = D1.astype(np.float32)
        self.D2 = D2.astype(np.float32)
        self.W1 = W1
        self.W2 = W2
        self.H = H
        self.rank = rank
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.geneset1 = geneset1
        self.geneset2 = geneset2
        self.cluster = cluster

    def gene_selection(self, rm_value1=2, rm_value2=0, threshold=0.06):
        """
        Gene filter same as SC3 clustering [Kiselev et al, 2017, nature methods(doi:10.1038/Nmeth.4236)].
        This function removes gene that are either expressed (expression value > rm_value1)
        in less than (threshold*100)% of cells (rare genes) or expressed (expression value > rm_value2)
        in at least (threshold*100)% of cells (ubiquitous genes).

        Parameters
        ----------
        rm_value1 : int or float, default 2
            threshold for counts as "gene expression" in removing rare genes.

        rm_value2 : int or float, default 0
            threshold for counts as "gene expression" in removing ubiquitous genes.

        threshold : float, default 0.06
            threshold of the number of cells that satisfy the condition for removing.


        """
        self.gene_set1 = list(set(self.D1.index[(self.D1 > rm_value1).sum(axis=1) > threshold * len(self.D1.columns)])
                              & set(self.D1.index[(self.D1 > rm_value2).sum(axis=1) < (1 - threshold) * len(self.D1.columns)]))
        self.gene_set2 = list(set(self.D2.index[(self.D2 > rm_value1).sum(axis=1) > threshold * len(self.D2.columns)])
                              & set(self.D2.index[(self.D2 > rm_value2).sum(axis=1) < (1 - threshold) * len(self.D2.columns)]))
        self.D1 = self.D1.loc[self.gene_set1, :]
        self.D2 = self.D2.loc[self.gene_set2, :]

    def log_scale(self):

        self.D1 = np.log2(self.D1 + 1)
        self.D2 = np.log2(self.D2 + 1)

    def normalize(self, norm='l1', normalize='cell'):
        """
        This function normalize the input gene expression data.

        Parameters
        ----------

        norm : str, default 'l1'
            Norm parameters for sklearn.preprocessing.Normalizer.

        normalize : str, defaault 'cell'
            Select 'cell' or 'gene' as the target of normalization.

        """

        norm = Normalizer(norm=norm, copy=False)
        if normalize == 'cell':
            self.D1 = norm.fit_transform(self.D1)
            self.D2 = norm.fit_transform(self.D2)
        elif normalize == 'gene':
            self.D1 = norm.fit_transform(self.D1.T).T
            self.D2 = norm.fit_transform(self.D2.T).T

    def factorize(self, solver='mu', init='random', device='gpu'):
        """
        This function fuctorize the input gene expession matrix as 'Joint-NMF'.

        Parameters
        ----------

        solver : str, default 'mu'
            The solver of Joint-NMF. In this version, only 'multiplicative update' is supported.

        init : str or None, defaut 'random'
            The initialization of factorized matrix. In this version, only 'random' is supported.

        device : str, default 'gpu'
            Select the device for matrix factorization. 'gpu' means it is calculated using GPU,
            and others means calculated using CPU.

        """

        print('start matrix factorization ......')
        if solver == 'mu':
            if init == 'random':
                self.W1, self.W2, self.H = random_init(
                    self.D1, self.D2, self.rank)
            elif self.W1 is None or self.W2 is None or self.H is None:
                print("select 'random' or set the value of factorized matrix.")

            if device == 'gpu':
                j_nmf = Joint_NMF_GPU(
                    self.D1, self.D2, self.W1, self.W2, self.H,
                    self.lambda1, self.lambda2, self.lambda3, self.lambda4,
                    iter_num=10000, conv_judge=1e-5, calc_log=[])
                self.W1, self.W2, self.H = j_nmf.calc()
            else:
                j_nmf = Joint_NMF_CPU(
                    self.D1, self.D2, self.W1, self.W2, self.H,
                    self.lambda1, self.lambda2, self.lambda3, self.lambda4,
                    iter_num=10000, conv_judge=1e-5, calc_log=[])
                self.W1, self.W2, self.H = j_nmf.calc()

        print('finished!!')

    def clustering(self, method='hierarchical', cluster_num=None):
        """
        This function classify the cells of input data.

        Parameters
        ----------

        method : str, default 'hierarchical'
            Select the methods for clustering. In this version, only 'Hierarchical clustering'.
            is supported.

        cluster_num : int or None, default None
            Give the number of clusters.
        """

        if method == 'hierarchical':
            self.cluster = linkage(self.H.T, method='ward')
            self.cluster = fcluster(self.cluster,
                                    t=cluster_num,
                                    criterion="maxclust")
