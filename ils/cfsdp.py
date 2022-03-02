'''
Clustering by Fast Search and Find of Density Peaks
Reference:
1. Clustering by fast search and find of density peaks
   https://science.sciencemag.org/content/344/6191/1492
2. Clustering  by Fast  Search  and  Find of  Density Peaks  with  Data  Field
   http://cje.ejournal.org.cn/en/article/doi/10.1049/cje.2016.05.001
'''

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from .basic_ils import ILS

def calculate_dc(distance_matrix, percentage=1.2):
    '''
    This function is to calulate the chosen cut-off distance.
    INPUTS:
        distance_matrix: N * N matrix (calculated by pairwise_distance)
        percentage: choose the precentage of the nearest point around each point of the total
    OUTPUTS:
        dc: cut-off distance
     '''
    temp = []
    # use the upper triangle number of the distance matrix
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            temp.append(distance_matrix[i][j])
    # sorted the list
    temp.sort()
    dc = temp[int(len(temp) * percentage / 100)]
    return dc

def continuous_density(distance_matrix, dc):
    '''
    Measure the density of a point by counting the number of points whose distance is less than dc
    INPUTS:
        distance_matrix: N * N matrix (calculated by pairwise_distance)
        dc: cut-off distance
    OUTPUTS:
        density: density of N points
    '''
    density = np.zeros(shape=len(distance_matrix))
    for index, dis in enumerate(distance_matrix):
        # guassian function
        density[index] = np.sum(np.exp(-(dis / dc) ** 2))
    return density

def discrete_density(distance_matrix, dc):
    '''
    Measure the density of a point by counting the number of points whose distance is less than dc
    INPUTS:
        distance_matrix: N * N matrix (calculated by pairwise_distance)
        dc: cut-off distance
    OUTPUTS:
        density: density of N points
    '''
    density = np.zeros(shape=len(distance_matrix))
    for index, dis in enumerate(distance_matrix):
        # the length of the points larger than the cut-off distance
        density[index] = len(dis[dis < dc])
    return density

def delta_function(distance_matrix, density):
    '''
    Calculate the nearest distance and the immediate superior of a point whose density is greater than its own.
    INPUTS:
        distance_matrix: N * N matrix (calculated by pairwise_distance)
        density: density of N points
    OUTPUTS:
        delta_matrix: computing the minimum distance between the point and any other with high density
        closest_point: the closest point with higher distance than the current point
    '''
    delta_matrix = np.zeros(shape=len(distance_matrix))
    closest_point = np.zeros(shape=len(distance_matrix), dtype=np.int32)
    for index, dis in enumerate(distance_matrix):
        # The set of points whose density is greater than the current point
        density_larger = np.squeeze(np.argwhere(density > density[index]))
        if density_larger.size != 0:
            distance_between_larger = distance_matrix[index][density_larger]
            delta_matrix[index] = np.min(distance_between_larger)
            min_distance_index = np.squeeze(np.argwhere(distance_between_larger == delta_matrix[index]))
            # If there are multiple points whose density is greater than oneself and which are closest to oneself,
            # select the first point as the immediate superior
            if min_distance_index.size >= 2:
                min_distance_index = np.random.choice(a=min_distance_index)
            if distance_between_larger.size > 1:
                closest_point[index] = density_larger[min_distance_index]
            else:
                closest_point[index] = density_larger
        # largest density point
        else:
            delta_matrix[index] = np.max(distance_matrix)
            closest_point[index] = index
    return delta_matrix, closest_point

def density_delta(density, delta_matrix, df):
    '''
    Draw the delta diagram and the raw data diagram(2D, need dimension reduction first).
    INPUTS:
        density: density of N points
        delta_matrix: computing the minimum distance between the point and any other with high density
        df: original dataframe(2D)
    OUTPUTS:
        density and delta plot
        original dataset plot
    '''
    plt.figure(figsize=(8, 6))
    ax1 = plt.subplot(121)
    for i in range(len(df)):
        plt.scatter(x=density[i], y=delta_matrix[i], c='k', marker='o', s=15)
    plt.xlabel('density')
    plt.ylabel('delta')
    plt.title('Decision gragh')
    plt.sca(ax1)

    ax2 = plt.subplot(122)
    plt.scatter(x=df['x'], y=df['y'],  marker='o', c='b', s=8)
    plt.xlabel('axis_x')
    plt.ylabel('axis_y')
    plt.title('Original_dataset')
    plt.sca(ax2)
    plt.show()

def choosing_centernumber(density, delta_matrix):
    '''
    Calculate the product of the density value and the minimum distance value at each point and draw the decision graph
    INPUTS:
        density: density of N points
        delta_matrix: computing the minimum distance between the point and any other with high density
    OUTPUTS:
        gamma: the product of the density value and the minimum distance value at each point
        sorted gamma graph
    '''
    # normalizate the data
    normal_density = (density - np.min(density)) / (np.max(density) - np.min(density))
    normal_delta = (delta_matrix - np.min(delta_matrix)) / (np.max(delta_matrix) - np.min(delta_matrix))
    gamma = normal_density * normal_delta
    plt.figure(num=2, figsize=(8, 6))
    plt.scatter(x=range(len(delta_matrix)), y=-np.sort(-gamma), c='k', marker='o', s=-np.sort(-gamma) * 100)
    plt.xlabel('data_num')
    plt.ylabel('gamma')
    plt.title('Gamma graph')
    plt.show()
    return gamma

def applyILS(X, X_embedded, index, colors):
    '''
    Combined the ILS with the centorid finding, and plot the 2D results
    INPUTS:
        X: the original dataset
        X_embedded: 2D dataframe, column name 'x' and 'y'
        index: the index list of the top k gamma values
        colors: color list defined before
    OUTPUTS:
        newL: new labels
        2D cluster plot
    '''
    df1 = X.copy()
    df1['label'] = 0
    count = 1
    for i in index:
        df1.loc[i, 'label'] = count
        count += 1
    print(count)
    print("The number of clusters: " + str(count - 1))
    df1.index.name = 'ID'
    features = [ i for i in df1.columns if i != 'label' ]
    newL, orderedL = ILS(df1[features + ['label']],'label')
    L = pd.DataFrame(newL)
    L.index = L.index.map(int)
    L.LS = L.LS.astype("int")
    X_embedded.index.name = 'ID'
    df2 = pd.merge(X_embedded, L, on = 'ID')
    plt.figure(figsize=(10, 8))
    for i in range(1, count):
        plt.scatter(df2[df2['LS'] == i]['x'], df2[df2['LS'] == i]['y'], c=colors[i], alpha=0.5)
    plt.show()
    return newL

def top_k_idx(gamma, k):
    '''
    Sorted the gamma list and get the top k values' indexes of the list
    INPUTS:
        gamma: the product of the density value and the minimum distance value at each point
        k: the top k number
    OUTPUT:
        index: the index list of the top k gamma values
    '''
    print(sorted(gamma,reverse=True)[:k])
    idx = gamma.argsort()
    idx = idx[::-1]
    index = idx[:k]
    return index