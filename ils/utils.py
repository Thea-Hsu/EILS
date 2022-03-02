from matplotlib import pyplot as plt
import numpy as np
from itertools import cycle, islice
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks_cwt

'''
Common functions for the whole package
'''
# colors = np.array(list(islice(cycle(
#         ['#837E7C','#377eb8',
#          '#4daf4a','#f781bf', '#a65628', '#ff7f00',
#          '#984ea3','#999999', '#e41a1c', '#dede00']
#          ),int(10))))

colors = np.array(list(islice(cycle(['#FAEBD7','#00FFFF','#7FFFD4','#F0FFFF','#F5F5DC',
                                    '#FFE4C4','#FFEBCD','#0000FF','#8A2BE2','#A52A2A',
                                    '#DEB887','#5F9EA0','#7FFF00','#D2691E','#FF7F50',
                                     '#6495ED','#FFF8DC','#DC143C','#00FFFF','#00008B']), int(20))))

def plot_ILSdistances(df, minR, centroid, label):
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

# plot local minima red, local maxima green, curve yellow, window size is the parameter
def findMin(vec, window):
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
def min_toCentroid(df, centroid = None , features = None ):
    '''
    INPUT:
        df = pandas dataFrame:
                columns are dimensions
        centroid = list or tuple with consistant dimension
        features = string or list of strings:
                select only these columns of df
    '''
    if type(features) == type(None) :
        features = df.columns
    if type(centroid) == type(None) :
        centroid = df[features].mean()
    dist = df.apply(lambda row : sum(
            [(row[j] - centroid[i])**2  for i, j in enumerate(features)]), axis = 1 )
    return dist.idxmin()


