# import libraries
from matplotlib import pyplot as plt
import time
import math
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn import cluster, datasets, mixture
from scipy.signal import find_peaks
from scipy.signal import find_peaks_cwt
from scipy.ndimage.filters import gaussian_filter1d
from itertools import cycle, islice
import warnings
warnings.filterwarnings("ignore")