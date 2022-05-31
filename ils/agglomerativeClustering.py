from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from sklearn.cluster import AgglomerativeClustering
from sklearn import decomposition as skldec
from matplotlib import pyplot as plt 
import random
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def get_linkage(dataset, method='single', metric='euclidean'):
    """Generate the agglometive linkage matrix.

    Args:
        Details: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#:~:text=The%20hierarchical%20clustering%20encoded%20as%20a%20linkage%20matrix.
        dataset (dataframe): dataset
        method (str): linkage methods. Defaults to 'single'.
        metric (str): pdist function. Defaults to 'euclidean'.

    Returns:
        ndarray: encoded linkage matrix
    """
    Z = linkage(dataset, method=method, metric=metric)
    return Z

def plot_dendrogram(Z, dataset, cutoff_dic):
    """Draw dendrogram to determine the upper bound of cut off height. 

    Args:
        Z (ndarray): encoded linkage matrix
        dataset (dataframe): dataset
        cutoff_dic (float): cut-off line
    """
    plt.figure(figsize=(20, 6),dpi=100)
    plt.xlabel("Index")
    plt.ylabel('Height')
    plt.grid(True)
    d = dendrogram(Z, labels = dataset.index,  truncate_mode='lastp', leaf_rotation=90, leaf_font_size=8)
    plt.axhline(y=cutoff_dic, c='k')

def cut_height(Z, height):
    """Cut-off the linkage matrix.

    Args:
        Z (ndarray): encoded linkage matrix
        height (float): the height to cut the tree = cutoff_dic in plot_dendrogram

    Returns:
        ndarray: an array indicate group membership
    """
    label = cut_tree(Z,height=height)
    label = label.reshape(label.size,)
    return label

def count_label(label):
    """Count the number of data point of each label.

    Args:
        label (ndarray): an array indicate group membership

    Returns:
        dictionary: dictionary of countable labels
    """
    dic = {}
    label = list(label)
    for k in label:
        dic[k] = dic.get(k, 0) + 1
    dic = dict(sorted(dic.items(), key=lambda item:item[1], reverse=True))
    return dic

def calculate_topklabel(label, k):
    """The top k labelled data proportion of the whole dataset.

    Args:
        label (ndarray): an array indicate group membership
        k (int): the number of top k pick up

    Returns:
        int: percentage
    """
    dic = count_label(label)
    count = 0
    values = list(dic.values())
    for i in range(k):
        count += values[i]
    per = count/len(label) * 100
    print(f'Top {k} label data is of {per}%')
    return per

def random_list(length=10):
    """Generate a random list for cut-off search

    Args:
        length (int, optional): random list list. Defaults to 10.

    Returns:
        list: a random number list belong to [0, 1]
    """
    if length >= 0:
        length=int(length)
        random_list = []
        for i in range(length):
            random_list.append(random.random())
    return random_list

def choose_best_cut(Z, k, low=60, high=95, random_list=random_list(10), random_search=True):
    """Using both random search and grid search for cut-offing.
    If one search method does not find any cut-off value, 
    then it will automaticly change to the other one.
    If both of the 20 values cannot searched, 
    please reconsider the k has been chosen,
    or regenerate the random list.

    Args:
        Z (ndarray): encoded linkage matrix
        k (int): the number of clusters
        low (int): lower bound
        high (int): higher bound
        random_list (list): a random number list belong to [0, 1]
        random_search (bool, optional): using random search or grid search. Defaults to True.

    Returns:
        ndarray: an array indicate group membership
    """
    best_score = 0
    cut_height = 0
    label_final = None
    grid_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if random_search == True:
        search_list = random_list
    else:
        search_list = grid_list
    for i in search_list:
        label = cut_tree(Z,height=i)
        label = label.reshape(label.size,)
        dic = count_label(label)
        count = 0
        values = list(dic.values())
        if len(values) < k:
            continue
        for j in range(k):
            count += values[j]
        per = count/len(label) * 100
        print(per)
        if per < high and per > low:
            if per > best_score:
                best_score = per
                print(best_score)
                cut_height = i
                label_final = label
    if best_score == 0:
        if search_list == random_list:
            search_list = grid_list
        else:
            search_list = random_list
        for i in search_list:
            label = cut_tree(Z,height=i)
            label = label.reshape(label.size,)
            dic = count_label(label)
            count = 0
            values = list(dic.values())
            if len(values) < k:
                continue
            for j in range(k):
                count += values[j]
            per = count/len(label) * 100
            print(per)
            if per < high and per > low:
                if per > best_score:
                    best_score = per
                    print(best_score)
                    cut_height = i
                    label_final = label
    else:
        pass
    print(f'Cut height: {i}')
    print(f'Top {k} label data is of {best_score}%')
    return label_final

def get_index(label):
    """Get the correspoding index list of agglomerative clustering.

    Args:
        label (ndarray): an array indicate group membership

    Returns:
        list: list of list
    """
    unique = list(set(label))
    arr = np.array(label)
    index_list = []
    for i in unique:
        index_list.append(list(np.where(arr==i)[0]))
        index_list = sorted(index_list, key=len, reverse=True)
    return index_list

def split_data(label, b):
    """Spliting the labelled data and unlabelled data.

    Args:
        label (ndarray): an array indicate group membership
        b (int): the number of clusters

    Returns:
        list: labelled with agglomertive clustering
        list: unlabelled data
    """
    categories = get_index(label)
    remain = []
    ac = []
    for i in range(b):
        ac.append(categories[i])
    e = len(get_index(label))
    for i in range(b, e):
        remain.append(categories[i])
    return ac, remain

def pca_vision(dataset, n_components, label, text=False):
    """Using PCA to visialize the label and dataset

    Args:
        dataset (dataframe): dataset
        n_components (float): number of components to keep
        label (ndarray): an array indicate group membership
        text (bool, optional): using random search or grid search. Defaults to False.
    """
    pca = skldec.PCA(n_components = n_components)    
    pca.fit(dataset)  
    result = pca.transform(dataset)  
    plt.figure() 
    plt.scatter(result[:, 0], result[:, 1], c=label, cmap=plt.get_cmap('gist_rainbow_r'), edgecolor='k') 
    if text == True:
        for i in range(result[:,0].size):
            plt.text(result[i,0], result[i,1], dataset.index[i], color='gray')    
    x_label = 'PC1(%s%%)' % round((pca.explained_variance_ratio_[0]*100.0),2)   
    y_label = 'PC1(%s%%)' % round((pca.explained_variance_ratio_[1]*100.0),2)   
    plt.xlabel(x_label)    
    plt.ylabel(y_label)    