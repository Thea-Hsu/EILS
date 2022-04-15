from matplotlib import pyplot as plt
import numpy as np
from sklearn import mixture
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings("ignore")

def plot_mixtures(n_compoments=30, dataset=None):
    """Using the sklearn BayesianGaussianMixture to give a expected mixture weight.

    Args:
        n_compoments (int, optional): The number of mixture components. Defaults to 30.
        dataset (dataframe): dataset. Defaults to None.

    Returns:
       number of weighted component: 95% component
    """
    DPGMM = mixture.BayesianGaussianMixture(n_components=n_compoments,
                                            max_iter=100000000,
                                            n_init=10,
                                            tol=1e-5,
                                            init_params='kmeans',
                                            weight_concentration_prior_type='dirichlet_process')
    DPGMM.fit(dataset)
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_w = np.arange(30) + 1
    ax.bar(plot_w - 0.5, np.sort(DPGMM.weights_)[::-1], width=1., lw=0)
    ax.set_xlim(0.5, 30)
    ax.set_xlabel('Component')
    ax.set_ylabel('Posterior expected mixture weight')
    return countComponent(DPGMM.weights_, 0.95)


def countComponent(weightList, ratio = 0.95):
    """Count the number of components

    Args:
        weightList (list): the weights of each components
        ratio (float, optional): cut-off ratio of weighted components. Defaults to 0.95.

    Returns:
        int: the number of weighted components
    """
    weightList = np.sort(weightList)[::-1]
    countWeight = 0
    for i in range(len(weightList)):
        if countWeight < ratio:
            countWeight += weightList[i]
        else:
            break
    print('Number of components: ' , i+1)
    