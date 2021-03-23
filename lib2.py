
import pandas as pd
import numpy as np
import pickle

def loadObject(sFileName):
    """
    Load python object from binary file
    
    Parameters
    ----------
    sFileName : string
        File name
        
    Returns
    -------
    obj : object
        Object loaded from binary file if successful, None otherwise
    """
    
    try:
        f = open(sFileName, "rb")
        obj = pickle.load(f)
        f.close()
    except:
        return None
    return obj

def getScore(y_true, y_pred, labelsDict=None):

    returnDict = {'Accuracy': None, 'Misclassification': None,\
            'Confusion_Matrix': None, 'Labels': None}
        
    tDf = pd.DataFrame(np.array(y_true).reshape(-1, 1), columns=['x1'])
    tDf['x2'] = np.array(y_pred).reshape(-1, 1)
    tDf.dropna(axis=0, how='any', inplace=True)

    y_true = np.array(tDf['x1'].values)
    y_pred = np.array(tDf['x2'].values)
    
    # y_true = np.array(y_true).reshape(-1)
    # y_pred = np.array(y_pred).reshape(-1)
    y = np.concatenate((y_true, y_pred))
    classLabels = np.unique(y) # sorted array
    if len(classLabels) < 2:
        print('Number of unique labels={}. Nothing to calculate'.format(len(classLabels)))
        return returnDict    
    
    # check labels
    if labelsDict is None:
        labelsDict = {}
    for c in classLabels:
        if c not in labelsDict.keys():
            labelsDict[c] = c
            
    index = []
    columns = []
    for label in labelsDict.keys():
        index.append('{}{}'.format('Predicted ', labelsDict[label]))
        columns.append('{}{}'.format('True ', labelsDict[label]))
    cm = np.zeros(shape=(len(classLabels), len(classLabels)), dtype=int)


    # reshape 2D array
    if y_true.ndim == 2:
        y_true = y_true.reshape(-1)
    if y_pred.ndim == 2:
        y_pred = y_pred.reshape(-1)
    if len(y_true) != len(y_pred):
        print('len of y_true != y_pred')
        return False
    nTotal = len(y_true)
    for i in range(0, len(y_true), 1):
        y_trueValue = y_true[i]
        y_predValue = y_pred[i]
        row = None
        column = None
        for idx, label in enumerate(classLabels, start=0):# tuples[index, label]
            if label == y_trueValue:
                column = idx
            if label == y_predValue:
                row = idx
        if row is None or column is None:
            print("Crash getScore. Class does not contain label")
            return False
        cm[row, column] += 1 
                    
    cmFrame = pd.DataFrame(cm, index=index, columns=columns, dtype=int)
# sum of all non-diagonal cells of cm / nTotal        
    misclassification = 0
    accuracy = 0
    for i in range(0, len(classLabels), 1):
        for j in range(0, len(classLabels), 1):
            if i == j:
                accuracy += cm[i, j]
                continue
            misclassification += cm[i, j]
    misclassification /= nTotal  
    accuracy /= nTotal 
    out = {}
    for i in range(0, len(classLabels), 1):
        out[labelsDict[classLabels[i]]] = {}
        if np.sum(cm[i, :]) == 0:
            out[labelsDict[classLabels[i]]]['Precision'] = 1
        else:
            out[labelsDict[classLabels[i]]]['Precision'] = cm[i, i] / np.sum(cm[i, :]) # tp / (tp + fp)
        if np.sum(cm[:, i]) == 0:
            out[labelsDict[classLabels[i]]]['Recall'] = 1
        else:
            out[labelsDict[classLabels[i]]]['Recall'] = cm[i, i] / np.sum(cm[:, i]) # tp / (tp + fn)
        if (out[labelsDict[classLabels[i]]]['Precision'] + out[labelsDict[classLabels[i]]]['Recall']) == 0:
            out[labelsDict[classLabels[i]]]['F1'] = 0
        elif np.isinf(out[labelsDict[classLabels[i]]]['Precision']) or np.isinf(out[labelsDict[classLabels[i]]]['Recall']):
            out[labelsDict[classLabels[i]]]['F1'] = 0
        else:
            out[labelsDict[classLabels[i]]]['F1'] = 2 * (out[labelsDict[classLabels[i]]]['Precision'] * \
                out[labelsDict[classLabels[i]]]['Recall']) / (out[labelsDict[classLabels[i]]]['Precision'] + \
                out[labelsDict[classLabels[i]]]['Recall'])

    return {'Accuracy': accuracy, 'Misclassification': misclassification,\
            'Confusion_Matrix': cmFrame, 'Labels': out}
    
def sortLists(list1, list2, reverse=False):
    zipped_lists = zip(list1.copy(), list2.copy())
    sorted_pairs = sorted(zipped_lists, reverse=reverse)
    tuples = zip(*sorted_pairs)
    list1, list2 = [list(tuple) for tuple in  tuples]
    return list1, list2

def sortLists3(list1, list2, list3, reverse=False):
    zipped_lists = zip(list1.copy(), list2.copy(), list3.copy())
    sorted_pairs = sorted(zipped_lists, reverse=reverse)
    tuples = zip(*sorted_pairs)
    list1, list2, list3 = [list(tuple) for tuple in  tuples]
    return list1, list2, list3  

def formatTime(t):
    d, reminder = divmod(int(t), 3600*24)
    h, reminder = divmod(reminder, 3600)
    m, s = divmod(reminder, 60)
    return d, h, m, s



