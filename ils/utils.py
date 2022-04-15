from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from itertools import cycle, islice
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks_cwt
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler

'''
Common functions for the whole package
'''
# colors = np.array(list(islice(cycle(
#         ['#837E7C','#377eb8',
#          '#4daf4a','#f781bf', '#a65628', '#ff7f00',
#          '#984ea3','#999999', '#e41a1c', '#dede00']
#          ),int(10))))#7FFFD4
# '#FAEBD7','#00FFFF','#c21535','#F0FFFF','#F5F5DC',
#                                     '#FFE4C4','#FFEBCD','#0000FF','#8A2BE2','#A52A2A',
#                                     '#DEB887','#5F9EA0','#7FFF00','#D2691E','#FF7F50',
#                                      '#6495ED','#FFF8DC','#DC143C','#00FFFF','#00008B
"""Define some colors list for ploting
"""
colors = np.array(list(islice(cycle(['#837E7C','#377eb8','#4daf4a','#f781bf', '#a65628', 
                                    '#ff7f00','#984ea3','#999999', '#e41a1c', '#dede00',
                                    '#DEB887','#5F9EA0','#7FFF00','#D2691E','#FF7F50',
                                    '#6495ED','#FFF8DC','#DC143C','#00FFFF','#00008B']), int(20))))

def plot_ILSdistances(df, minR, centroid, label):
    """Ordered ILS labelling distances (Rmin), and cluster plots
    A side by side of the ordered Rmin plot and the points in the cluster with the centroid identified (in 2D). 
    For a 2D feature space this allows a comparison of the defined cluster and the connection to details of the Rmin plot 
    (Only the latter might be available in a higher dimensional feature space).

    Args:
        df (dataframe): dataset
        minR (Series): ordered Rmin 
        centroid (2d point): the mean value of the features
        label (int): ILS label
    """
    fig = plt.figure(figsize=(6,3))
    fig.subplots_adjust(left=.07, right=.98, bottom=.001,
                        top=.96, wspace=.05,hspace=.01)

    ax = plt.subplot(1, 2, 1)
    plt.ylim(0, 1)
    plt.xticks(()); plt.yticks(())
    ax.plot(range(len(minR)), minR, color=colors[label])

    ax = plt.subplot(1, 2 , 2)
    plt.xticks(()); plt.yticks(())
    plt.xlim(-3,3); plt.ylim(-3, 3)
    ax.scatter(df['x'].values, df['y'].values, s=4, color=colors[0])
    ax.scatter(centroid[0], centroid[1], s=3,
               color=colors[label],marker = 'x', linewidth =20)

def findMin(vec, window):
    """plot local minima red, local maxima green, curve yellow, window size is the parameter

    Args:
        vec (Series): Rmin
        window (scalar): standard deviation for Gaussian kernel

    Returns:
        minima, maxima: minima list and maxima list
    """
    filtered = gaussian_filter1d(vec, window)
    index = np.arange(len(filtered))

    maxima = find_peaks_cwt(filtered, len(filtered) * [window])
    maxima = [i for i in maxima if i < len(filtered) - window]
    maxima = [i for i in maxima if i > window]

    betweenMax = np.split(filtered, maxima)
    betweenIndex = np.split(index, maxima)

    minVal = [min(i) for i in betweenMax]
    subMinIndex = [np.argmin(i) for i in betweenMax]
    minima = [betweenIndex[i][subMinIndex[i]] for i in range(len(subMinIndex))]
    minima = [i for i in minima if i != 0]

    fig = plt.figure()
    plt.plot(vec, '-', color='yellow')
    for i in minima:
        plt.plot(i, vec[i], 'ro')
    for i in maxima:
        plt.plot(i, vec[i], 'go')

    minima = [i for i in minima if i != 0]
    return minima, maxima

# choose the point nearest to the mean centroid
def min_toCentroid(df, centroid = None , features = None):
    """The closest point that exists in a data set to a general point 

    Args:
        df (dataframe): columns are dimensions
        centroid (list or tuple, optional): list or tuple with consistant dimension. Defaults to None.
        features (string or list of strings, optional): select only these columns of df. Defaults to None.

    Returns:
        idmin (Series): The closest point id in the data set
    """
    if type(features) == type(None) :
        features = df.columns
    if type(centroid) == type(None) :
        centroid = df[features].mean()
    dist = df.apply(lambda row : sum(
            [(row[j] - centroid[i])**2  for i, j in enumerate(features)]), axis = 1 )
    return dist.idxmin()


def synthetic_data(N=1500):
    """2D dummy examples

    Args:
        N (int, optional): number of instances. Defaults to 1500.

    Returns:
        X: A list of synthetic data, use index to get each type of dataset
    """
    noisy_circles = datasets.make_circles(n_samples=N, factor=.5, noise=.05, random_state=42)
    noisy_moons = datasets.make_moons(n_samples=N, noise=.05, random_state=42)
    blobs = datasets.make_blobs(n_samples=N, random_state=42)
    no_structure = np.random.rand(N, 2), None
    # Anisotropicly distributed data
    RS = 170 
    X, y = datasets.make_blobs(n_samples=N, random_state= RS)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=N,
                                 cluster_std=[1.0, 2.5, 0.5],
                                 random_state=RS)

    # data including x,y points and true labels
    ds = [ noisy_circles, noisy_moons, varied, aniso, blobs, no_structure ] 

    # scale and store points only in a list of dataFrames 
    features = ['x','y']
    X = []
    for i,j in enumerate(ds) :
        X.append(pd.DataFrame(StandardScaler().fit_transform(j[0])
                    ,columns = features) )
        X[i].index.name = 'ID'

    return X