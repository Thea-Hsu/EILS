
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

def ILS(df, labelColumn, outColumn='LS', iterative=True):
    '''
    @author: amanda.parker@data61.csiro.au
    Citation and implemenatation details : ""

    Apply iterative label spreading in a multi-dimensional feature-space.
    Returns labels for all points and the order-labelled
    and distance-when-labelled for all newly labelled points.
    INPUTS :
        df = pandas dataFrame:
            all features are columns (and only those) +
            one column holding initial labels
        iterative = boolean :
            True : label spreading to unlabelled points applied iteratively
            False : all unlabelled points relabelled with regard to
                    initially labelled set
        labelColumn = String:
            Column name for column that holds initial labels.
            0 = to be labelled
            positive integers = assigned label.
    OUTPUTS :
        pandas dataSeries:
            index : same index input df
            name : outColumn
            data : Labels for all points
                (all values 0 replaced with a positive integer)
        pandas dataFrame:
            Only contains points that were labelled by ILS
            index : same as input df *reordered by order labelled*
            columns:
                minR : distance when relabelled
                IDclosestLabelled : ID of point label recieved from'
     '''

    featureColumns = [i for i in df.columns if i != labelColumn]
    # Keep original index columns in DF
    indexNames = list(df.index.names)
    oldIndex = df.index
    df = df.reset_index(drop=False)

    # separate labelled and unlabelled points
    labelled = [
        group for group in df.groupby(df[labelColumn] != 0)
    ][True][1].fillna(0)
    unlabelled = [
        group for group in df.groupby(df[labelColumn] != 0)
    ][False][1]

    # lists for ordered output data
    outD = []
    outID = []
    closeID = []

    # Continue while any point is unlabelled
    while len(unlabelled) > 0:
        # Calculate labelled to unlabelled distances matrix (D)
        D = pairwise_distances(
            labelled[featureColumns].values,
            unlabelled[featureColumns].values)

        # Find the minimum distance between a labelled and unlabelled point
        # first the argument in the D matrix
        (posL, posUnL) = np.unravel_index(D.argmin(), D.shape)
        # then convert to an index ID in the data frame
        # (The ordering will switch during iterations, more robust)
        idUnL = unlabelled.iloc[posUnL].name
        idL = labelled.iloc[posL].name

        # Switch label from 0 to new label
        unlabelled.loc[idUnL, labelColumn] = labelled.loc[idL, labelColumn]
        # move newly labelled point to labelled dataframe
        labelled = labelled.append(unlabelled.loc[idUnL])
        # labelled.loc[list(unlabelled.loc[idUnL])[0]] = list(unlabelled.loc[idUnL])
        # drop from unlabelled data frame
        unlabelled.drop(idUnL, inplace=True)

        # output the distance and id of the newly labelled point
        outD.append(D.min())
        outID.append(idUnL)
        closeID.append(idL)

    # Throw error if loose or duplicate points
    if len(labelled) + len(unlabelled) != len(df):
        raise Exception(
            '''The number of labelled ({}) and unlabelled ({}) 
                points does not sum to the total ({})'''.format(
                len(labelled), len(unlabelled), len(df)))

    # Reodered index for consistancy
    newIndex = oldIndex[outID]

    orderLabelled = pd.Series(
        data=outD, index=newIndex, name='minR')
    # ID of point label was spread from
    closest = pd.Series(
        data=closeID, index=newIndex, name='IDclosestLabel')
    labelled = labelled.rename(columns={labelColumn: outColumn})
    # new labels as dataseries
    newLabels = labelled.set_index(indexNames)[outColumn]

    # return
    return newLabels, pd.concat([orderLabelled, closest], axis=1)
