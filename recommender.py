# General recommender

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import lib4
import lib2
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
#from deepctr.feature_column import VarLenSparseFeat
from tensorflow.keras.metrics import AUC
from tensorflow.keras.metrics import BinaryAccuracy
import time
from tensorflow.keras.callbacks import EarlyStopping
from deepctr.models import DeepFM
from deepctr.models.wdl import WDL
from datetime import timedelta
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from deepctr.models.nfm import NFM
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scikitplot.metrics import plot_roc_curve
import random
from datetime import date

###############################################################################
#%%
def list_get(l, idx, default):
    if idx < len(l) and idx >= 0:
        return l[idx]
    return default

#%%
def makePredictions(recommendationsDictList, mapUserToItem, mapUserToItemPrior, \
    itemToDescDict, nRecommendations=5):
    """
    mapUserToItem = mapUserToLic
    mapUserToItemPrior = mapUserToItemPrior
    nRecommendations=5
    itemToDescDict=mapLicToDesc
    """

    users = []
    itemsTrue = []
    itemsPrior = []
    itemsPredictNested = []
    scoresPredictNested = []
    
    nModels = len(recommendationsDictList)
    usersList = []
    for d in recommendationsDictList:
        users1 = [x for x in d.keys()]
        usersList.extend(users1)
        itemsPredictNested.append([])
        scoresPredictNested.append([])
    usersList = sorted(list(set(usersList)))
    
    for user in usersList:
#       user = 937
        itemsTrue1 = mapUserToItem.get(user, set())
        itemsTrue1 = sorted(list(itemsTrue1))
        itemsPrior1 = mapUserToItemPrior.get(user, set())
        itemsPrior1 = sorted(list(itemsPrior1))

        nRecords = max(nRecommendations, len(itemsTrue1))
        
        itemsNested1 = []
        scoresNested1 = []
        for d in recommendationsDictList:
            d1 = d.get(user, {})
            itemsPredict1 = d1.get('items', [])
            scoresPredict1 = d1.get('scores', [])
            itemsNested1.append(itemsPredict1)
            scoresNested1.append(scoresPredict1)
       
        for i in range(0, nRecords, 1):
            users.append(user)
            itemTrue = list_get(itemsTrue1, i, '')            
            itemsTrue.append(itemTrue)
            if itemTrue in itemsPrior1:
                itemsPrior.append(True)
            else:
                itemsPrior.append(False)

            for j in range(0, len(itemsNested1), 1):
                itemsPredictNested[j].append(list_get(itemsNested1[j], i, ''))
                scoresPredictNested[j].append(list_get(scoresNested1[j], i, 0))
            
    data = pd.DataFrame(columns=['user', 'itemPrior', 'itemTrue', 'itemTrueDesc'])

    data['user'] = users
    data['itemPrior'] = itemsPrior
    data['itemTrue'] = itemsTrue
    data['itemTrueDesc'] = data['itemTrue'].map(itemToDescDict)
    data['itemTrueDesc'] = data['itemTrueDesc'].fillna('')
    
    for i in range(1, nModels+1, 1):
        columnItem = 'item_{}'.format(i)
        columnScore = 'score_{}'.format(i)
        columnDesc = 'desc_{}'.format(i)
        data[columnScore] = scoresPredictNested[i-1]
        data[columnItem] = itemsPredictNested[i-1]
        data[columnDesc] = data[columnItem].map(itemToDescDict)
        data[columnDesc] = data[columnDesc].fillna('')

    return data

#%%
def getScoresCF(userToPredictDict, mapUserToItemPosterior, itemsList, n=None):
    
    if n is None:
        n = np.inf
        
    nMax = 20
    itemsSet = set(itemsList)
    scores = {}
    scores['accuracy'] = {}    
    scores['jaccard'] = {}
    scores['cm'] = {}
    scores['precision'] = {}
    scores['recall'] = {}
    scores['f1'] = {}
    for i in range(1, nMax+1, 1):
        scores['accuracy'][i] = []
        scores['jaccard'][i] = []
        scores['cm'][i] = []
        scores['precision'][i] = []
        scores['recall'][i] = []
        scores['f1'][i] = []   
        
    for user, valueDict in userToPredictDict.items():

        trueItemsPosterior1 = mapUserToItemPosterior.get(user, set())

        mPosterior = len(trueItemsPosterior1)        
        if mPosterior == 0:
            continue # nothing to compare
        
        if len(trueItemsPosterior1.intersection(itemsSet)) == 0:
            continue # none of posterior item belongs to itemsSet

        nLoop = min(mPosterior, n)
        nLoop = min(nLoop, nMax)
        
#        if mPosterior > nMax:
#            print(user, trueItemsPosterior1)
#            raise ValueError('Crap 2')
            

        for nRecommendations in range(1, nLoop+1, 1):
            if nRecommendations > len(trueItemsPosterior1):
                break
            topRecomItems = set(valueDict['items'][0:nRecommendations])
            
            nIntersection = len(trueItemsPosterior1.intersection(topRecomItems))
            nUnion = len(trueItemsPosterior1.union(topRecomItems))

            if nUnion > 0:
                jaccard = nIntersection / nUnion
            else:
                jaccard = 0
            
            tp = len(topRecomItems.intersection(trueItemsPosterior1))
            fp = nRecommendations - tp
            tmp1 = itemsSet.difference(trueItemsPosterior1)
            tmp2 = itemsSet.difference(topRecomItems)
            tn = len(tmp1.intersection(tmp2))
            fn = mPosterior - tp
            
            cmDf = pd.DataFrame([[tn, fn],[fp, tp]], index=['Predicted_0', 'Predicted_1'],\
                columns=['Actual_0', 'Actual_1'], dtype=int)
                        
            if (tp + fn) == 0:
                recall = 0
            else:
                recall = tp / (tp + fn)

            if (tp + fp) == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)
                
            if (precision + recall) == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            scores['accuracy'][nRecommendations].append(accuracy)
            scores['jaccard'][nRecommendations].append(jaccard)
            scores['cm'][nRecommendations].append(cmDf)                                
            scores['precision'][nRecommendations].append(precision)                
            scores['recall'][nRecommendations].append(recall)
            scores['f1'][nRecommendations].append(f1)
                    
    return scores

#%%
def recommendationsToDict(userItemSorted, userItemScores):
    """
    """
    
    mapUserToPredict = {}
    for user in userItemSorted.index:
        mapUserToPredict[user] = {}
        mapUserToPredict[user]['items'] = list(userItemSorted.loc[user])
        mapUserToPredict[user]['scores'] = list(userItemScores.loc[user])

    return mapUserToPredict

#%%
def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

#%%
def getAccuracyRecent(model, modelName, test, feature_names, batch_size, 
                      resultsDir='', threshold=0.5, plot=True):
    
    testCopy = test.copy(deep=True)
    test_model_input = {name:testCopy[name] for name in feature_names}
    pred_proba = model.predict(test_model_input, batch_size=batch_size)
    testCopy['proba'] = pred_proba
    test1 = testCopy[testCopy['prior'] == 0].copy(deep=True)
    predict = np.where(test1['proba'] > threshold, 1, 0)
    pred_proba = test1['proba'].values
    response = test1['response'].values
    if len(np.unique(response)) < 2:
        print('Nothing to score')
        return None, None, None, None, None

    pred_proba = pred_proba.reshape(-1)
    pred_proba0 = 1 - pred_proba
    forPlot = np.concatenate((pred_proba0.reshape(-1, 1), pred_proba.reshape(-1, 1)), axis=1)

    score05 = lib2.getScore(response, predict, labelsDict={0: 'No', 1: 'Buy'})
    
    tNp = np.arange(0.01, 1, 0.01)
    tNp = [round(x, 4) for x in tNp]
    tNp.extend([0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999])
    f1List = []
    scores = []
    for t in tNp:
        predict = np.where(test1['proba'] > t, 1, 0)
        score = lib2.getScore(response, predict, labelsDict={0: 'No', 1: 'Buy'})
        scores.append(score)
        if score.get('Labels') is None:
            f1List.append(0)
        else:
            f1List.append(score['Labels']['Buy']['F1'])
        
    idx = argmax(f1List)
        
    nUnique = test1['response'].nunique()
    if nUnique > 1:
        rocAuc = roc_auc_score(test1['response'].values, test1['proba'].values)
    else:
        rocAuc = np.nan
        
    if plot:
        plot_roc_curve(np.array(test1['response'].values).reshape(-1, 1), forPlot)
        plt.savefig(os.path.join(resultsDir, 'ROC_{}.png'.format(modelName)), bbox_inches='tight')
        plt.close() 

    return test1, score05, rocAuc, tNp[idx], scores[idx]

#%%
def fitModel(model, data, inp, metrics, nEpochs=10, batch_size=128, patience=10, \
        validation_split=0.2, class_weight=None):
    start_time = time.time()

    model.compile("adam", "binary_crossentropy", metrics=metrics)

    if patience is not None:
        early_stopping = EarlyStopping(monitor='val_auc', verbose=1, patience=patience,
            mode='max', restore_best_weights=True) # val_auc    
        earlyStoping = True
    else:
        earlyStoping = False
        
    if earlyStoping:
        history = model.fit(inp, data['response'].values, batch_size=batch_size,\
            epochs=nEpochs, validation_split=validation_split, shuffle=True,\
            class_weight=class_weight, callbacks=[early_stopping])
    else:
        history = model.fit(inp, data['response'].values, batch_size=batch_size,\
            epochs=nEpochs, validation_split=validation_split, shuffle=True,\
            class_weight=class_weight)
                 
    end_time = time.time()
    t = int(round(end_time - start_time, 0))
    return model, history, t

#%%
def plotHistory(history, fileName=None):        
    # plot progress
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model')
    plt.ylabel('Y')
    plt.xlabel('epoch')
    plt.legend(['auc', 'val_auc', 'accuracy', 'val_accuracy'])
    if fileName is None:
        plt.show()     
    else:
        plt.savefig(fileName, bbox_inches='tight')
        plt.close() 
    return

#%%
def mergeRobust(data1, data2, on, how, fillValue=0):
    
    dTypesDict = {}
    
    def getDict(data, dTypesDict={}):
        for c, d in zip(data.columns, data.dtypes):
            dTypesDict[c] = str(d)
        return dTypesDict
        
    dTypesDict = getDict(data1, dTypesDict={})
    dTypesDict = getDict(data2, dTypesDict=dTypesDict)

#    print(dTypesDict, '\n')
    
    data = data1.merge(data2, on=on, how=how)
    for column, value in dTypesDict.items():
        if value == 'uint32':
            data[column] = data[column].fillna(fillValue)
#            print(column, value)
            data[column] = data[column].astype('uint32')
    
    if np.sum(data.isna().sum()) > 0:
        raise RuntimeError('NaN in data')

    return data

#%%
def encode(data, column, default='_', model=None, encoder='robust', error='raise'):
    if column not in data.columns:
        if error == 'raise':
            raise ValueError('column {} not in data.columns'.format(column))
        else:
            return data
        
    dataCopy = data.copy(deep=True)
    if model is None:
        if encoder == 'dummy':
            enc = DummyEncoderRobust(unknownValue=default)
            dataCopy = enc.fit_transform(dataCopy, column, prefix=None)
        elif encoder == 'robust':
            enc = LabelEncoderRobust(unknownValue=default)
            x = list(dataCopy[column].values)
        elif encoder == 'sklearn':
            dataCopy[column] = dataCopy[column].fillna(default)            
            enc =  LabelEncoder()
            x = list(dataCopy[column].values)            
        elif encoder == 'minMax':
            dataCopy[column] = dataCopy[column].fillna(default)
            enc = MinMaxScaler(feature_range=(0, 1))
            x = np.array(dataCopy[column].values).reshape(-1, 1)
        elif encoder == 'standard':
            dataCopy[column] = dataCopy[column].fillna(default)            
            enc = StandardScaler(copy=True, with_mean=True, with_std=True)
            x = np.array(dataCopy[column].values).reshape(-1, 1)

        if encoder in ['robust', 'sklearn', 'minMax', 'standard']:
            dataCopy[column] = enc.fit_transform(x)

        return dataCopy, enc
    else:
        if encoder in ['robust', 'sklearn', 'minMax', 'standard']:
            dataCopy[column] = model.transform(dataCopy[column].values)

        elif encoder == 'dummy':
            dataCopy = enc.transform(dataCopy, column, prefix=None)
        return dataCopy
    return

#%%
def fillFeatures(data, indexList, mapFeatureToType, mapFeatureToMissing, 
    mapMissingToDefault):
    """
    indexList join data on those idx and fill gaps
    """
    
#        drop duplicates, fill nan
    if not isinstance(data, pd.DataFrame):
        raise RuntimeError('transactions must be pandas DataFrame')            
#        drop duplicated users
    dataCopy = data[~data.index.duplicated(keep='first')].copy(deep=True)
#        add users that appeared in transactions but not in usersFeatures data
    if indexList is None:
        indexList = list(dataCopy.index)
    df2 = pd.DataFrame(index=indexList)
    dataCopy = dataCopy.join(df2, how='right')
#   fill missing values        
    for feature, featureType in mapFeatureToType.items():
        missing = mapFeatureToMissing.get(feature)
        if missing is None:
            missing = mapMissingToDefault.get(featureType)
                
        if feature in dataCopy.columns:
            dataCopy[feature] = dataCopy[feature].fillna(missing)

    return dataCopy 
    
#%%
def encodeFeatures(data, mapFeatureToType, mapFeatureToMissing, mapMissingToDefault, 
    sparseType='label', encoderDict=None):
    """
    sparseType in ['label', 'dummy']
    dense and sparse for now
    
    varlen to do
    
    """
    
    if encoderDict is None:
        returnEncoder = True
        encoderDict = {}
    else:
        returnEncoder = False
        
    dataCopy = data.copy(deep=True)
    
    for f, v in mapFeatureToType.items():
        
        if f not in dataCopy.columns:
            continue
        
        if v == 'sparse':
            default = mapFeatureToMissing.get(f, mapMissingToDefault['sparse'])
            if sparseType == 'label':
                encoder = 'robust'
            elif sparseType == 'dummy':
                encoder = 'dummy'
            else:
                raise ValueError('Wrong argument sparseType = {} for sparse feature'.format(sparseType))
                
        elif v == 'dense':
            encoder = 'minMax'
            default = mapFeatureToMissing.get(f, mapMissingToDefault['dense'])
                     
        else:
            raise ValueError('Feature type {} is not supported yet'.format(v))
            
        if returnEncoder:
            dataCopy, le = encode(dataCopy, f, default, model=None, encoder=encoder)
            if (v == 'sparse') and (sparseType == 'label'):            
                dataCopy[f] = dataCopy[f].astype('uint32')            
            encoderDict[f] = le
        else:
            model = encoderDict.get(f)
            if model is None:
                raise ValueError('No encoder for feature {}'.format(f))
            dataCopy = encode(dataCopy, f, default, model=encoderDict[f], encoder=encoder)
            if (v == 'sparse') and (sparseType == 'label'):            
                dataCopy[f] = dataCopy[f].astype('uint32')

    if returnEncoder:
        return dataCopy, encoderDict
    return dataCopy
    
#%%
class DummyEncoderRobust():
    
    """
    fit_transform - call LabelEncoderRobust first
    add prefix == column name
    add dummy prefix_
    
    transform - encode with LabelEncoderRobust instance
    make dummy
    check created dummy and add missing and sort
    """
    
    def __init__(self, unknownValue='_', error='raise'):
# unknownValue always has label 0        
        self.unknownValue = unknownValue
#        self.mapKeyToLabel = None
        self.error = error    
        
#        fit_transform
        self.le = None
        self.column = None
        self.prefix = None
        self.columnsDummy = None

#%%        
    def fit_transform(self, data, column, prefix=None):
        self.column = column
        if prefix is None:
            prefix = str(self.column)
        self.prefix = prefix
        
        dataCopy = data.copy(deep=True)
        self.le = LabelEncoderRobust(unknownValue=self.unknownValue, error=self.error)
        y = self.le.fit_transform(dataCopy[self.column])
        newData = pd.DataFrame(index=dataCopy.index)
        newData[self.column] = y        

        dummyDf = pd.get_dummies(newData[self.column], prefix=prefix, prefix_sep='_',\
            dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)
        self.columnsDummy = list(dummyDf.columns)
        
#        if no unknown values, reserve column
        firstColumn = '{}_0'.format(prefix)
        if firstColumn not in self.columnsDummy:
            dummyDf[firstColumn] = 0
            dummyDf[firstColumn] = dummyDf[firstColumn].astype('uint8')
            self.columnsDummy.insert(0, firstColumn)
            dummyDf = dummyDf[self.columnsDummy]

        dataCopy.drop(columns=[self.column], inplace=True)
        dataCopy = pd.concat([dataCopy, dummyDf], axis=1)
        return dataCopy

#%%
    def transform(self, data):
        if self.le in None:
            print('fit_transform first')
            return

        if self.column not in data.columns:
            raise ValueError("Column '{}' not in data.columns".format(self.column))
            return
        
        dataCopy = data.copy(deep=True)
        y = self.le.transform(dataCopy[self.column])
        newData = pd.DataFrame(index=dataCopy.index)
        newData[self.column] = y

        dummyDf = pd.get_dummies(newData[self.column], prefix=self.prefix, prefix_sep='_',\
            dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)

        columnsDummy = list(dummyDf.columns)
        
        for c in self.columnsDummy:
            if c not in columnsDummy:
                dummyDf[c] = 0
                dummyDf[c] = dummyDf[c].astype('uint8')                
        
        dummyDf = dummyDf[self.columnsDummy]

        dataCopy.drop(columns=[self.column], inplace=True)
        dataCopy = pd.concat([dataCopy, dummyDf], axis=1)
        return dataCopy
    
#%%
###############################################################################        
#%%
class LabelEncoderRobust():
    """
    # unknownValue always has label 0
    """
    
    def __init__(self, unknownValue='_', error='raise'):
        self.unknownValue = unknownValue
        self.mapKeyToLabel = None
        self.error = error
        
#%%    
    def fit(self, x):
        x = list(x)
        s = set(x)
        if self.unknownValue in s:
            x = list(s)
            x.remove(self.unknownValue)
            x.insert(0, self.unknownValue)
        else:
            x = list(s)
            x.insert(0, self.unknownValue)
            
        self.classes_ = x
        self.nLabels = len(x)
        self.keys = x
        self.labels = list(np.arange(0, self.nLabels, 1))
        self.mapKeyToLabel = {}
        for i in range(0, self.nLabels, 1):
            self.mapKeyToLabel[self.keys[i]] = self.labels[i]

        self.mapLabelToKey = {}
        for key, label in self.mapKeyToLabel.items():
            self.mapLabelToKey[label] = key
            
#%%
    def transform(self, x):
        if self.mapKeyToLabel is None:
            if self.error == 'raise':
                raise ValueError('Fit first')
            elif self.error == 'warn':
                print('Fit first')
                return None
        
        labels = []
        for key in x:
            label = self.mapKeyToLabel.get(key, 0)
            labels.append(label)
            
        return labels
    
#%%             
    def inverse_transform(self, labels):
        if self.mapLabelToKey is None:
            if self.error == 'raise':
                raise ValueError('Fit first')
            elif self.error == 'warn':
                print('Fit first')
                return None        
        
        keys = []
        for label in labels:
            key = self.mapLabelToKey.get(label, self.unknownValue)
            keys.append(key)
            
        return keys        
    
#%%
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
#%%
###############################################################################
#%%
class Recommender(object):
    """
    Timeline models:
        
        1. General
            daysToBuyPrior = None
            or
            reference = None
            daysToBuyPosterior, dLeftMargin, dRightMargin become None
            
        2. First purchase
            reference = 'first buy'
            daysToBuyPrior : int
            daysToBuyPosterior : int or None
            dLeftMargin, dRightMargin become None
            
        3. Left margin
            reference = 'left'
            dLeftMargin : date
            dRightMargin becomes None
            daysToBuyPrior : int
            daysToBuyPosterior : int or None
            
        4. Right margin
            reference = 'right'
            dLeftMargin : becomes None
            dRightMargin : date or None (will be today)
            daysToBuyPrior : int
            daysToBuyPosterior : int or None   
            
-------------------------------------------------------------------------------

    Data required for recommender:
        transactions : pandas DataFrame, columns = ['user', 'item', 'date_t']
    
        userFeatures : pandas DataFrame, index == users from 'user' column in transactions
        mapUserFeatureToType : dict, key == column name in userFeatures, value in ['dense', 'sparse', 'varlen']
        mapUserFeatureToDefault : dict, key == column name in userFeatures, value - default value to replace NaN with
        mapMissingToDefault : dict, example: {'sparse': '_', 'dense': 0, 'varlen': ''} values might be different, keys are constant
        
    Methods:
        __init__(self, useUsers=True, useItems=True, seed=None)
        loadTransactions()

    """

    def __init__(self, usersTrain='all', itemsTrain='all', usersRecommend='recent', 
          itemsRecommend='all', trainPosterior=True, daysToBuyPrior=30, 
          daysToBuyPosterior=365, reference='first buy', margins=(None, None), resultsDir=''):
        """
        usersRecommend in ['recent', 'random'] or list of users
        """
        
        super().__init__()
                
        self.trainPosterior = trainPosterior
        
        if isinstance(usersTrain, list):
            self.users = [str(x) for x in usersTrain]
        else:
            self.users = None
            
        if isinstance(itemsTrain, list):
            self.items = [str(x) for x in itemsTrain]
        else:
            self.items = None            

        if isinstance(itemsRecommend, list):
            self.itemsRecommend = [str(x) for x in itemsRecommend]
        else:
            self.itemsRecommend = None 

        if isinstance(usersRecommend, list):
            self.usersRecommend = [str(x) for x in usersRecommend]
        else:
            self.usersRecommend = usersRecommend              

        self.daysToBuyPrior = daysToBuyPrior
        self.reference = reference
        self.daysToBuyPosterior = daysToBuyPosterior
        if self.daysToBuyPosterior is None:
            self.daysToBuyPosterior = 0
            
        if self.daysToBuyPrior is None or self.reference is None:
#            print('Tag 1')
#            general case, use all data
            self.daysToBuyPrior = 365000
            self.daysToBuyPosterior = 0
            self.reference = 'None'
            self.dLeftMargin = date(1900, 1, 1)
            self.dRightMargin = date(2100, 1, 1)            
                
        if self.reference == 'first buy':
            self.dLeftMargin = date(1900, 1, 1)
            self.dRightMargin = date(2100, 1, 1)
        else:
            self.dLeftMargin = margins[0]
            self.dRightMargin = margins[1]            
                        
        if self.reference == 'left' and self.dLeftMargin is None:
            raise ValueError("using reference 'left' assign date for left margin")
            
        if self.reference == 'right' and self.dRightMargin is None:
            self.dRightMargin = date.today()
            
        self.metrics = [BinaryAccuracy(name='accuracy'), AUC(name='auc')]
        self.resultsDir = resultsDir
        self.columnsTransactions = ['user', 'item', 'date_t']    
        
#        loadTransactions()
        self.transactions = None
        self.mapUserToItem = None # dict, user -> set(purchased items)
        self.firstTransaction = None
        self.lastTransaction = None

#       loadTransactions() -> getUserToPurchases()
        self.mapUserToPurchases = None
        self.mapUserToDateToItems = None
        
#       loadTransactions() ->  getUserToMarginBuy()
        self.mapUserToFirstBuy = None
        self.mapUserToLastBuy = None
        
#       loadTransactions() ->  getUserToItemPriorPosterior()
        self.mapUserToItemPrior = None
        self.mapUserToItemPosterior = None
        
#       loadTransactions() -> makeUserItemBinaryMatrix()
        self.mapUserToIdx = {}  
        self.mapItemToIdx = {} 
        self.userItemBinaryMatrix = None
        
#        makeItemItemJaccardMatrix()
        self.itemItemJaccardMatrix = None
        
#        loadUserFeatures()  
        self.mapUserFeatureToType = None
        self.mapUserFeatureToDefault = None
        self.mapUserMissingToDefault = None
        self.userFeatureLabelDf = None
        self.userEncoderLabelDict = None
        self.userFeatureDummyDf = None
        self.userEncoderDummyDict = None
        
#        loadItemFeatures()
        self.mapItemFeatureToType = None
        self.mapItemFeatureToDefault = None
        self.mapItemMissingToDefault = None
        self.itemFeatureLabelDf = None
        self.itemEncoderLabelDict = None
        
#        makeUserItemTrain()
        self.dataTrain = None

#        makeUserItemRecommend()        
        self.dataRecommend = None
                
#%%
    def loadTransactions(self, data):
        """
        data : pandas DataFrame, must contain ['user', 'item', 'date_t']
        
        Create:
            self.daysToBuyPrior if None -> 36500
            self.daysToBuyPosterior if None -> 36500
            self.transactions
            self.usersList
            self.itemsList
            self.mapUserToItem            
        """
        
        if not isinstance(data, pd.DataFrame):
            raise RuntimeError('transactions must be pandas DataFrame')
        if len(set(self.columnsTransactions).intersection(set(data.columns))) != 3:
            raise RuntimeError("transactions must contain columns: ['user', 'item', 'date_t']")        

        self.transactions = data[self.columnsTransactions].copy(deep=True)
        self.transactions.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        self.transactions.drop_duplicates(subset=self.columnsTransactions,  keep='first', inplace=True)
        self.transactions['user'] = self.transactions['user'].astype(str)
        self.transactions['item'] = self.transactions['item'].astype(str)
        
        users = list(self.transactions['user'].unique())
        if self.users is not None:
            self.users = sorted(list(set(self.users).intersection(users)))
        else:
            self.users = sorted(users)
            
        items = list(self.transactions['item'].unique())
        if self.items is not None:
            self.items = sorted(list(set(self.items).intersection(items)))
        else:
            self.items = sorted(items)
            
        if self.itemsRecommend is not None:
            self.itemsRecommend = sorted(list(set(self.itemsRecommend).intersection(self.items)))
        else:
            self.itemsRecommend = sorted(self.items)
                        
        self.firstTransaction = self.transactions['date_t'].min()
        self.lastTransaction = self.transactions['date_t'].max()
        
        df1 = self.transactions.groupby('user')['item'].apply(set).reset_index(name='itemsSet')
        df1.set_index('user', drop=True, inplace=True)
        self.mapUserToItem = df1['itemsSet'].to_dict()
                
        self.getUserToPurchases()
        self.getUserToMarginBuy()
        self.getUserToItemPriorPosterior()
        self.checkUsersRecommend()
        self.makeUserItemBinaryMatrix()
              
#%%
    def getUserToPurchases(self):
        """
        Create:
            self.mapUserToPurchases dict, user -> ([date1, date2], [itemslist1, itemslist2])
            self.mapUserToDateToItems dict, user -> valueDict: key == date, 
                value == list of items purchased at that day not repeating items
        """
        
        def f(x):
            return list(x['date_t']), list(x['itemsList'])
        
        g10 = self.transactions.groupby(['user', 'date_t'])['item'].apply(list).reset_index(name='itemsList')
        g10['sItemSorted'] = g10['itemsList'].apply(lambda x: ','.join(sorted(list(set(x)))))
        g10['sItemSorted'] = g10['sItemSorted'].str.split(',')
        g10.sort_values(['user', 'date_t'], inplace=True)
        g10['date_t'] = g10['date_t'].dt.date
        g11 = g10.groupby(['user']).apply(f)
        self.mapUserToPurchases = g11.to_dict()
            
        self.mapUserToDateToItems = {}
        for user in self.mapUserToPurchases.keys():
            dates, items = self.mapUserToPurchases.get(user)
            
            prevItems = []
            dateDict = {}
            
            for i in range(0, len(dates), 1):
                items1 = items[i]
                if i == 0:
                    prevItems.extend(items1)
                    dateDict[dates[i]] = items1
                else:
                    for item in items1:
                        if item not in prevItems:
                            prevItems.append(item)
                            old = dateDict.get(dates[i], [])
                            old.append(item)
                            dateDict[dates[i]] = old
    
            self.mapUserToDateToItems[user] = dateDict
        
#%%
    def getUserToMarginBuy(self):
        """
        Create:
            self.mapUserToFirstBuy
            self.mapUserToLastBuy
        """
        g = self.transactions.groupby(['user']).agg({'date_t' : ['first', 'last']}).reset_index()
        g.columns = ['user', 'first', 'last'] 
        g.set_index('user', inplace=True)
        self.mapUserToFirstBuy = g['first'].to_dict()
        self.mapUserToLastBuy = g['last'].to_dict()
                
#%%
    def getUserToItemPriorPosterior(self):
        """
        Create:
            self.mapUserToItemPrior : user -> items set bounded by [first buy date .. first buy date + daysToBuyPrior]
            self.mapUserToItemPosterior : user -> items set bounded by [first buy date + daysToBuyPrior + 1 .. first buy date + daysToBuyPrior + 1 + 365]            
    
        """
        self.mapUserToItemPrior = {}
        self.mapUserToItemPosterior = {}        

#       general case; everything is prior, no posterior
        if self.reference == 'None':
            for user in self.mapUserToPurchases.keys():
                _, itemsNestedList = self.mapUserToPurchases[user]
                itemsList = []
                for i in itemsNestedList:
                    itemsList.extend(i)                    
                self.mapUserToItemPrior[user] = set(itemsList)
                self.mapUserToItemPosterior[user] = set()            
 
        elif self.reference == 'first buy':
            for user in self.mapUserToPurchases.keys():
                dStartPrior = self.mapUserToFirstBuy.get(user, None)
                dEndPrior = dStartPrior + timedelta(days=self.daysToBuyPrior)
                dStartPosterior = dEndPrior + timedelta(days=1)
                dEndPosterior = dStartPosterior + timedelta(days=self.daysToBuyPosterior)
                dates, items = self.mapUserToPurchases.get(user)
                itemsPrior = []
                itemsPosterior = []
                for i in range(0, len(dates), 1):
                    d = dates[i]
                    if (d >= dStartPrior) and (d <= dEndPrior):
                        itemsPrior.extend(items[i])
                    elif (d >= dStartPosterior) and (d <= dEndPosterior):
                        itemsPosterior.extend(items[i])                        
                self.mapUserToItemPrior[user] = set(itemsPrior)
                self.mapUserToItemPosterior[user] = set(itemsPosterior)
            
        elif self.reference in ['left', 'right']:
            if self.reference == 'left':
                dStartPrior = self.dLeftMargin
                dEndPrior = dStartPrior + timedelta(days=self.daysToBuyPrior)
                dStartPosterior = dEndPrior + timedelta(days=1)
                dEndPosterior = dStartPosterior + timedelta(days=self.daysToBuyPosterior)
            elif self.reference == 'right':
                dEndPosterior = self.dRightMargin
                dStartPosterior = dEndPosterior - timedelta(days=self.daysToBuyPosterior)
                if self.daysToBuyPosterior > 0:
                    dEndPrior = dStartPosterior - timedelta(days=1)
                else:
                    dEndPrior = self.dRightMargin
                dStartPrior = dEndPrior - timedelta(days=self.daysToBuyPrior)
                            
                for user in self.mapUserToPurchases.keys():
                    dates, items = self.mapUserToPurchases[user]
                    itemsPrior = []
                    itemsPosterior = []            
                    for i in range(0, len(dates), 1):
                        d = dates[i]
                        if (d >= dStartPrior) and (d <= dEndPrior):
                            itemsPrior.extend(items[i])
                        elif (d >= dStartPosterior) and (d <= dEndPosterior) and (self.daysToBuyPosterior > 0):
                            itemsPosterior.extend(items[i])                        
                    self.mapUserToItemPrior[user] = set(itemsPrior)
                    self.mapUserToItemPosterior[user] = set(itemsPosterior)
                        
        else:
            raise ValueError('Value {} for reference is not supported'.format(self.reference))                        
            
#%%                                    
    def checkUsersRecommend(self):            
        if isinstance(self.usersRecommend, list):
            self.usersRecommend = sorted(list(set(self.usersRecommend).intersection(self.users)))
        elif isinstance(self.usersRecommend, str):
            if self.usersRecommend == 'recent':                
                self.leftMarginFirstBuy = self.lastTransaction - timedelta(days=self.daysToBuyPrior)
                usersRecommend = []
                for user, firstBuy in self.mapUserToFirstBuy.items():
                    if firstBuy >= self.leftMarginFirstBuy:
                        usersRecommend.append(user)
                self.usersRecommend = usersRecommend
            else:
                raise ValueError('Only supported recent for now')
        elif isinstance(self.usersRecommend, int):
            if self.usersRecommend > int(0.5 * len(self.users)):
                raise ValueError('Number of users to recommend {} is too big comparing to number of all users {}'.format(len(self.usersRecommend), len(self.users)))
            self.usersRecommend = random.sample(self.users, self.usersRecommend)            
            
        if len(self.usersRecommend) > 10000:
            raise ValueError('Number of users to recommend {} is too big. Max value = 10000'.format(len(self.usersRecommend)))
            
        self.usersRecommend = [str(x) for x in self.usersRecommend]
        
#%%        
    def makeUserItemBinaryMatrix(self):
        """
        Create:
            self.userItemBinaryMatrixDf
        """

        for idx, user in enumerate(self.users):
            self.mapUserToIdx[user] = idx

        for idx, item in enumerate(self.items):
            self.mapItemToIdx[item] = idx            
            
    #   make binary user-item matrix    
        self.userItemBinaryMatrix = np.zeros(shape=(len(self.users), len(self.items)), dtype=bool)
        usersRecommendSet = set(self.usersRecommend)
        for user in self.users:
            if user in usersRecommendSet:
#   only items purchased in prior period                
                items1 = self.mapUserToItemPrior.get(user, set())
            else:
#   items purchased in prior or posterior period                
                itemsPrior1 = self.mapUserToItemPrior.get(user, set())
                itemsPosterior1 = self.mapUserToItemPosterior.get(user, set())
                if self.trainPosterior:
                    items1 = itemsPrior1.union(itemsPosterior1)
                else:
                    items1 = itemsPrior1
                    
            for item in items1:                
                i = self.mapUserToIdx[user]
                j = self.mapItemToIdx[item]
                self.userItemBinaryMatrix[i, j] = True

#%%
    def userUserRecommend1(self, nRecommendations=None, rescale='no'):
        """
        rescale in ['no', 'all', 'each']

        Return:
            userItemSortedDf, userItemScoresDf            
        """
        
        if nRecommendations is None:
            nRecommendations = np.inf            
        
        usersRecommendIdx = []
        for user in self.usersRecommend:
            idx = self.mapUserToIdx[user]
            usersRecommendIdx.append(idx)
        
        itemsRecommendIdx = []
        for item in self.itemsRecommend:
            idx = self.mapItemToIdx[item]
            itemsRecommendIdx.append(idx)
            
        x1 = self.userItemBinaryMatrix[usersRecommendIdx, :]
#        get user user similarity according to purchase paterns
        userUserCosNp = cosine_similarity(self.userItemBinaryMatrix, x1, dense_output=True)
    
        userItemScoresNp = np.zeros(shape=(len(self.usersRecommend), len(self.itemsRecommend)), dtype='float32')
        for i in range(0, len(self.usersRecommend), 1):
            user = self.usersRecommend[i]
            itemsPrior = self.mapUserToItemPrior.get(user, set())
            a = userUserCosNp[:, i]
            for j in range(0, len(self.itemsRecommend), 1):
                item = self.itemsRecommend[j]
                if item in itemsPrior:
                    score = 0
                else:
                    itemIdx = itemsRecommendIdx[j]
                    b = self.userItemBinaryMatrix[:, itemIdx]
                    score = np.multiply(a, b).sum()
                userItemScoresNp[i, j] = score
    
    #    rescale to [0..1]        
        if rescale == 'all':
            maxScore = userItemScoresNp.max()
            userItemScoresNp = np.divide(userItemScoresNp, maxScore)
        elif rescale == 'each':
            userItemScoresNp = np.divide(userItemScoresNp, userItemScoresNp.max(axis=1).reshape(-1, 1))
                
    #    sort
        userItemSortedDf = pd.DataFrame(index=self.usersRecommend, columns=list(range(0, len(self.itemsRecommend), 1)))
        zeros = np.zeros(shape=(len(self.usersRecommend), len(self.itemsRecommend)), dtype='float32')
        userItemScoresDf = pd.DataFrame(zeros, index=self.usersRecommend, columns=list(range(0, len(self.itemsRecommend), 1)))
        
        for i in range(0, len(self.usersRecommend), 1):
            items = self.itemsRecommend.copy()
            score1, items1 = lib2.sortLists(list(userItemScoresNp[i, :]), items, reverse=True)
            user = self.usersRecommend[i]
            userItemSortedDf.loc[user] = items1
            userItemScoresDf.loc[user] = score1
    
#   cut        
        if nRecommendations < userItemSortedDf.shape[1]:
            columns = userItemSortedDf.columns[0:nRecommendations]
            userItemSortedDf = userItemSortedDf[columns]
            userItemScoresDf = userItemScoresDf[columns]
    
        return userItemSortedDf, userItemScoresDf

#%%         
    def makeItemItemJaccardMatrix(self):
        """
        score - P(j|i)  
        diagonal = 0
        
        Create:
            self.itemItemJaccardMatrix    
        """

        nItems = len(self.items)
        
        self.itemItemJaccardMatrix = np.zeros(shape=(nItems, nItems), dtype='float32')
        for i in range(0, nItems, 1):
            denominator = self.userItemBinaryMatrix[:, i].sum()
            for j in range(0, nItems, 1):
                if i == j:
                    continue
                intersection = np.where(((self.userItemBinaryMatrix[:, i] == True) & \
                    (self.userItemBinaryMatrix[:, j] == True)), 1, 0)
                if denominator == 0:
                    jaccard = 0
                else:
                    jaccard = np.sum(intersection) / denominator
    
                self.itemItemJaccardMatrix[i, j] = jaccard

#%%
    def itemItemRecommend(self, nRecommendations=None, rescale='no'):
        """
        rescale in ['no', 'all', 'each']
        
        Return:
            userItemSortedDf, userItemScoresDf                
        """
        
        self.makeItemItemJaccardMatrix()
        
        if nRecommendations is None:
            nRecommendations = np.inf
            
        itemsRecommendSet = set(self.itemsRecommend)
        
    #    prepare output frames
        userItemSortedDf = pd.DataFrame(index=self.usersRecommend, columns=list(range(0, len(self.itemsRecommend), 1)))
        zeros = np.zeros(shape=(len(self.usersRecommend), len(self.itemsRecommend)), dtype='float32')
        userItemScoresDf = pd.DataFrame(zeros, index=self.usersRecommend, columns=list(range(0, len(self.itemsRecommend), 1)))
        
        for user in self.usersRecommend:
            itemsPrior = self.mapUserToItemPrior.get(user, [])
            if len(itemsPrior) == 0:
#           user does not have purchase history, cannot use recommender
                continue
            
            itemsIdx = []
            for item in list(itemsPrior):
                itemsIdx.append(self.mapItemToIdx[item])
                
            rows = self.itemItemJaccardMatrix[itemsIdx, :]
            scores = list(rows.sum(axis=0))

#           discard predictions for existing purchases
            for i in range(0, len(scores), 1):
                item = self.items[i]
                if item in itemsPrior:
                    scores[i] = 0
            
            scores, items  = lib2.sortLists(scores, self.items, reverse=True)
    
    #   select only items in itemsRecommendSet
            scoresNew = []
            itemsNew = []
            for score, item in zip(scores, items):
                if item in itemsRecommendSet:
                    scoresNew.append(score)
                    itemsNew.append(item)
    
            userItemSortedDf.loc[user] = itemsNew
            userItemScoresDf.loc[user] = scoresNew
    
    #    rescale to [0..1]        
        if rescale == 'all':
            maxScore = np.max(userItemScoresDf.values)
            userItemScoresDf = userItemScoresDf.div(maxScore)
        elif rescale == 'each':
            maxScore = userItemScoresDf.max(axis=1)
            userItemScoresDf = userItemScoresDf.div(maxScore, axis=0)
    
        return userItemSortedDf, userItemScoresDf           
    
#%%    
    def loadUserFeatures(self, userFeatures, mapUserFeatureToType, 
                         mapUserFeatureToDefault, mapUserMissingToDefault):
        """
        Create:
            self.mapUserFeatureToType
            self.mapUserFeatureToDefault
            self.mapUserMissingToDefault
            self.userFeatureDummyDf
            self.userEncoderDummyDict
            self.userFeatureLabelDf
            self.userEncoderLabelDict

        """

        if not isinstance(userFeatures, pd.DataFrame):
            raise RuntimeError('transactions must be pandas DataFrame')
        if self.transactions is None:
            raise RuntimeError('Call loadTransactions first')
        
        self.mapUserFeatureToType = mapUserFeatureToType
        self.mapUserFeatureToDefault = mapUserFeatureToDefault
        self.mapUserMissingToDefault = mapUserMissingToDefault
        
        dataCopy = userFeatures.copy(deep=True)
        dataCopy = fillFeatures(dataCopy, indexList=self.users, \
            mapFeatureToType=self.mapUserFeatureToType,\
            mapFeatureToMissing=self.mapUserFeatureToDefault, \
            mapMissingToDefault=self.mapUserMissingToDefault)                  
        
        self.userFeatureDummyDf, self.userEncoderDummyDict = encodeFeatures(dataCopy, 
            self.mapUserFeatureToType, self.mapUserFeatureToDefault, 
            self.mapUserMissingToDefault, sparseType='dummy', encoderDict=None)        

        dataCopy['user'] = dataCopy.index
        dataCopy['userOrig'] = dataCopy.index
#        self.mapUserFeatureToType['user'] = 'sparse'
#        self.mapUserFeatureToDefault['user'] = '_'
        dataCopy.reset_index(drop=True, inplace=True)

        self.userFeatureLabelDf, self.userEncoderLabelDict = encodeFeatures(dataCopy, 
            self.mapUserFeatureToType, self.mapUserFeatureToDefault, 
            self.mapUserMissingToDefault, sparseType='label', encoderDict=None)
        
#%%
    def loadItemFeatures(self, itemFeatures, mapItemFeatureToType, 
        mapItemFeatureToDefault, mapItemMissingToDefault):
        """
        Create:
            self.mapItemFeatureToType
            self.mapItemFeatureToDefault
            self.mapItemMissingToDefault
            self.itemFeatureLabelDf
            self.itemEncoderLabelDict
        """
        
        if not isinstance(itemFeatures, pd.DataFrame):
            raise RuntimeError('transactions must be pandas DataFrame')
        if self.transactions is None:
            raise RuntimeError('Call loadTransactions first')
        
        self.mapItemFeatureToType = mapItemFeatureToType
        self.mapItemFeatureToDefault = mapItemFeatureToDefault
        self.mapItemMissingToDefault = mapItemMissingToDefault

        dataCopy = itemFeatures.copy(deep=True)
        dataCopy = fillFeatures(dataCopy, indexList=None, \
            mapFeatureToType=self.mapItemFeatureToType,\
            mapFeatureToMissing=self.mapItemFeatureToDefault, \
            mapMissingToDefault=self.mapItemMissingToDefault) 

        dataCopy['item'] = dataCopy.index
        dataCopy['itemOrig'] = dataCopy.index
#        self.mapItemFeatureToType['item'] = 'sparse'
#        self.mapItemFeatureToDefault['item'] = '_'
        dataCopy.reset_index(drop=True, inplace=True)
        
        self.itemFeatureLabelDf, self.itemEncoderLabelDict = encodeFeatures(dataCopy, 
            self.mapItemFeatureToType, self.mapItemFeatureToDefault, 
            self.mapItemMissingToDefault, sparseType='label', encoderDict=None)
        
#%%
    def userUserRecommend2(self, nRecommendations=None, rescale='no'):
        """
        rescale in ['no', 'all', 'each']

        Return:
            userItemSortedDf, userItemScoresDf                        
        """
        
        if nRecommendations is None:
            nRecommendations = np.inf
            
        userItemDf = pd.DataFrame(self.userItemBinaryMatrix, index=self.users, columns=self.items, dtype=bool)
        
        usersCommon = sorted(list(set(self.users).intersection(set(self.userFeatureDummyDf.index))))
#   some fresh users might be absent in customers data base, discard them
        usersRecommend = sorted(list(set(self.usersRecommend).intersection(set(usersCommon))))
#        same users
        userItemDf = userItemDf.loc[usersCommon]   
        userItemDf = userItemDf[self.itemsRecommend]
        userFeatureMatrix2Df = self.userFeatureDummyDf.loc[usersCommon]
        userFeatureMatrixRecommendDf = self.userFeatureDummyDf.loc[usersRecommend]
        
#        get user user similarity according to purchase paterns
        userUserCosNp = cosine_similarity(userFeatureMatrix2Df.values, 
            userFeatureMatrixRecommendDf.values, dense_output=True)
        userUserCosDf = pd.DataFrame(userUserCosNp, index=usersCommon,
            columns=usersRecommend)
    
        zeros = np.zeros(shape=(len(usersRecommend), len(self.itemsRecommend)), dtype='float32')
        userItemScoresDf = pd.DataFrame(zeros, index=usersRecommend, columns=self.itemsRecommend)
        for user in usersRecommend:
            itemsPrior = self.mapUserToItemPrior.get(user, set())     
            a = userUserCosDf[user].values
            for item in self.itemsRecommend:
                if item in itemsPrior:
                    score = 0  # ignore recommending purchased items
                else:                    
                    b = userItemDf[item].values
                    score = np.multiply(a, b).sum()
                userItemScoresDf[item].loc[user] = score
    
    #    rescale to [0..1]        
        if rescale == 'all':
            maxScore = np.max(userItemScoresDf.values)
            userItemScoresDf = userItemScoresDf.div(maxScore)
        elif rescale == 'each':
            userItemScoresDf = userItemScoresDf.div(userItemScoresDf.max(axis=0))
                
    #    sort
        userItemSortedDf = pd.DataFrame(zeros, index=usersRecommend, columns=list(range(0, len(self.itemsRecommend), 1)))
        for user in usersRecommend:
            items = self.itemsRecommend.copy()
            score1, items1 = lib2.sortLists(list(userItemScoresDf.loc[user].values), items, reverse=True)
            userItemSortedDf.loc[user] = items1
            userItemScoresDf.loc[user] = score1
    
#   cut
        if nRecommendations < userItemSortedDf.shape[1]:
            columns = userItemSortedDf.columns[0:nRecommendations]
            userItemSortedDf = userItemSortedDf[columns]
            userItemScoresDf = userItemScoresDf[columns]
    
        userItemScoresDf.columns = list(range(0, len(self.itemsRecommend), 1))
        return userItemSortedDf, userItemScoresDf
        

#%%
    def makeUserItemTrain(self):
        """
        Create:
            self.dataTrain
            self.mapFeatureToDefault
        """
        
        self.mapFeatureToDefault = {}
        for key, value in self.mapUserFeatureToDefault.items():
            self.mapFeatureToDefault[key] = value
        for key, value in self.mapItemFeatureToDefault.items():
            self.mapFeatureToDefault[key] = value
        
        if self.userFeatureLabelDf is None or self.itemFeatureLabelDf is None:
            raise RuntimeError('Run loadUserFeatures and loadItemFeatures first')
            
        users = []
        items = []
        response = []
        prior = []
            
        for user in self.users:
            
            itemsPrior1 = self.mapUserToItemPrior.get(user, set())
            itemsPosterior1 = self.mapUserToItemPosterior.get(user, set())
            if self.trainPosterior:
                items1 = itemsPrior1.union(itemsPosterior1)
            else:
                items1 = itemsPrior1

            for item in self.items:
                
                if item in items1:
                    response.append(1)
                else:
                    response.append(0)
                
                users.append(user)
                items.append(item)
                
                if item in itemsPrior1:
                    prior.append(1)
                else:
                    prior.append(0)
                                         
        for user in self.usersRecommend:
            itemsPrior1 = self.mapUserToItemPrior.get(user, set())
            for item in self.itemsRecommend:
                if item in itemsPrior1:
                    response.append(1)
                    prior.append(1)
                    users.append(user)
                    items.append(item)
                    
        self.dataTrain = pd.DataFrame(columns=['userOrig', 'itemOrig', 'prior', 'response'])
            
        self.dataTrain['userOrig'] = users
        self.dataTrain['itemOrig'] = items
        self.dataTrain['prior'] = prior
        self.dataTrain['response'] = response
        
        self.dataTrain = mergeRobust(self.dataTrain, self.userFeatureLabelDf, 
            on='userOrig', how='left', fillValue=0)

        self.dataTrain = mergeRobust(self.dataTrain, self.itemFeatureLabelDf, 
            on='itemOrig', how='left', fillValue=0)            

#%%
    def makeUserItemRecommend(self):
        """
        Create:
            self.dataRecommend
        """
    
        users = []
        items = []
        prior = []
        response = []
        for user in self.usersRecommend:            
            itemsPrior1 = self.mapUserToItemPrior.get(user, set())
            itemsPosterior1 = self.mapUserToItemPosterior.get(user, set()) # usefull for testing         
            for item in self.itemsRecommend:
                if item in itemsPrior1:
                    continue                
                users.append(user)
                items.append(item)
                prior.append(0)
                
                if item in itemsPosterior1:
                    response.append(1)
                else:
                    response.append(0)
                                
        self.dataRecommend = pd.DataFrame(columns=['userOrig', 'itemOrig', 'prior', 'response'])
            
        self.dataRecommend['userOrig'] = users
        self.dataRecommend['itemOrig'] = items
        self.dataRecommend['prior'] = prior
        self.dataRecommend['response'] = response

        self.dataRecommend = mergeRobust(self.dataRecommend, self.userFeatureLabelDf, 
            on='userOrig', how='left', fillValue=0)
        
        self.dataRecommend = mergeRobust(self.dataRecommend, self.itemFeatureLabelDf, 
            on='itemOrig', how='left', fillValue=0)
        
#%%
    def annRecommend(self, modelName, embedding_dim=10, dropout=0.5, nEpochs=1000, patience=10,\
        batch_size=1024, validFrac = 0.2, plot=False, getScore=False):
    
        """
        modelName : ['WDL', 'DeepFM', 'NFM']
        """
        
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.nEpochs = nEpochs
        self.patience = patience
        self.batch_size = batch_size
        self.validFrac = validFrac
        
        self.dataTrain = self.dataTrain.sample(frac=1)
        self.class_weight = lib4.getClassWeights(self.dataTrain['response'].values)
        
        dense_features = []
        sparse_features = []
        varlen_features = [] # to do
        
        for feature, t in self.mapUserFeatureToType.items():
            if t == 'dense':
                dense_features.append(feature)
            elif t == 'sparse':
                sparse_features.append(feature)
            elif t == 'varlen':
                varlen_features.append(feature)    
           
        for feature, t in self.mapItemFeatureToType.items():
            if t == 'dense':
                dense_features.append(feature)
            elif t == 'sparse':
                sparse_features.append(feature)
            elif t == 'varlen':
                varlen_features.append(feature) 
    
        encoderDict = {}
        for key, value in self.userEncoderLabelDict.items():
            encoderDict[key] = value
    
        for key, value in self.itemEncoderLabelDict.items():
            encoderDict[key] = value
    
        fixlen_sparse_feature_columns = [SparseFeat(feat, vocabulary_size=encoderDict[feat].nLabels+1,\
            embedding_dim=embedding_dim) for i, feat in enumerate(sparse_features)]
        
        fixlen_dense_feature_columns = [DenseFeat(feat, 1,) for feat in dense_features]
            
        linear_feature_columns = fixlen_sparse_feature_columns + fixlen_dense_feature_columns
        dnn_feature_columns = fixlen_sparse_feature_columns
    
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        
        train_model_input = {name:self.dataTrain[name] for name in feature_names}
        
        model = WDL(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(128, 128),\
            l2_reg_linear=1e-05, l2_reg_embedding=1e-05, l2_reg_dnn=0, seed=1024, \
            dnn_dropout=dropout, dnn_activation='relu', task='binary')
     
        model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', \
            dnn_dropout=dropout, dnn_hidden_units=(128, 128), l2_reg_linear=1e-05,\
            l2_reg_embedding=1e-05, l2_reg_dnn=0, seed=101, dnn_activation='relu',\
            dnn_use_bn=False)
        
        model = NFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(128, 128),\
            l2_reg_embedding=1e-05, l2_reg_linear=1e-05, l2_reg_dnn=0, seed=101, \
            bi_dropout=dropout, dnn_dropout=dropout, dnn_activation='relu', task='binary')
             
        model, history, timeElapsed = fitModel(model, self.dataTrain, train_model_input,\
            metrics=self.metrics, nEpochs=nEpochs, batch_size=batch_size,\
            patience=patience, validation_split=validFrac, class_weight=self.class_weight)
        
        if plot:
            plotHistory(history, os.path.join(self.resultsDir, 'history NFM.png'))    
              
        testCopy = self.dataRecommend.copy(deep=True)
        test_model_input = {name:testCopy[name] for name in feature_names}
        pred_proba = model.predict(test_model_input, batch_size=self.batch_size)
        testCopy['proba'] = pred_proba
        
        g = testCopy.groupby(['userOrig']).agg({'item': 'count', 'proba': ['min', 'max']})#.reset_index()
        g.columns = g.columns.droplevel(0)
        g['prior'] = g.index.map(self.mapUserToItemPrior)
        g['posterior'] = g.index.map(self.mapUserToItemPosterior)
        maxLen = g['count'].max()
        
        self.statsForecast = g
        
        tmp = testCopy[['userOrig', 'itemOrig', 'proba']].set_index('userOrig', drop=True).copy(deep=True)
        users = sorted(list(tmp.index.unique()))
    
        userItemSortedDf4 = pd.DataFrame(index=users, columns=list(range(0, maxLen, 1)))
        userItemScoresDf4 = pd.DataFrame(index=users, columns=list(range(0, maxLen, 1)))
    
        for user in users:
            tmp1 = tmp.loc[user]
            items1 = list(tmp1['itemOrig'].values)
            scores1 = list(tmp1['proba'].values)
            scores1, items1 = lib2.sortLists(scores1, items1, reverse=True)
            
            nExtend = maxLen - len(items1)
            scoresExtend = [0 for x in range(0, nExtend, 1)]
            itemsExtend = ['' for x in range(0, nExtend, 1)]
            scores1.extend(scoresExtend)
            items1.extend(itemsExtend)
    
            userItemSortedDf4.loc[user] = items1
            userItemScoresDf4.loc[user] = scores1   
        
        if getScore:
            test3, score3, rocAuc3, t3, score3t = getAccuracyRecent(model, modelName, 
                testCopy, feature_names, self.batch_size, resultsDir=self.resultsDir,
                threshold=0.5, plot=True)

            self.rocAuc = rocAuc3
            self.score = score3
            self.thresholdBest = t3
            self.scoreBest = score3t
            
        return userItemSortedDf4, userItemScoresDf4

#%%
#!!! rebuild and check
    def encodeVarlen(self, mapVarlenToTokenizerParams={}, mapVarlenToParams={}):

        def encodeFeature(data):   
            dataCopy = data.copy(deep=True)
            params = self.mapVarlenToParams.get(feature, {})
            tokenizer = Tokenizer(**tokenizerParams)
            text = list(dataCopy[feature].values)
            tokenizer.fit_on_texts(text)
            labels = tokenizer.texts_to_sequences(text)
            length = np.array(list(map(len, labels)))
            pad_size = min(max(length), params.get('max_len', 100000))
            params['pad_size'] = pad_size

            params['vocabSize'] = len(tokenizer.word_index) + 1
        
            textNp = pad_sequences(labels, maxlen=pad_size, padding='post')
            prefix = params.get('prefix', feature)
            columns = ['{}_{}'.format(prefix, i) for i in range(0, pad_size, 1)]            
            padDf = pd.DataFrame(textNp, index=dataCopy.index, columns=columns)
    
            dataCopy = pd.concat([dataCopy, padDf], axis=1)
            del(dataCopy[feature])
            params['pad_columns'] = columns
            params['tokenizer'] = tokenizer                
            self.mapVarlenToParams[feature] = params
            return dataCopy
        
        for f in [self.transactions, self.userFeatures, self.itemFeatures]:
            if f is None:
                raise RuntimeError('Call loadTransactions, loadUserFeatures, loadItemFeatures first')
        
        self.mapVarlenToTokenizerParams = mapVarlenToTokenizerParams
        self.mapVarlenToParams = mapVarlenToParams
        
        for feature, tokenizerParams in mapVarlenToTokenizerParams.items():
            if feature in self.userFeatures:
                self.userFeatures = encodeFeature(self.userFeatures)
            elif feature in self.itemFeatures:
                self.itemFeatures = encodeFeature(self.itemFeatures)            
        

#%%    
#   
#    def setModels(self, models):
#        """
#        modelsDict - dict, key == modelName, value == model instance
#        """
#        modelsDict = {'WDL': (WDL, {'dnn_hidden_units': (128, 128),\
#            'l2_reg_linear': 1e-05, 'l2_reg_embedding': 1e-05, 'l2_reg_dnn': 0,\
#            'dnn_dropout': 0.5, 'dnn_activation': 'relu', 'task': 'binary'}), 
#            'DeepFM': (DeepFM, {'task': 'binary', 'dnn_dropout': 0.5,\
#            'dnn_hidden_units': (128, 128), 'l2_reg_linear': 1e-05,\
#            'l2_reg_embedding': 1e-05, 'l2_reg_dnn': 0, 'dnn_activation': 'relu', 'dnn_use_bn': False}),\
#            'DCN': (DCN, {'cross_num': 2, 'cross_parameterization': 'vector',\
#            'dnn_hidden_units': (128, 128), 'l2_reg_linear': 1e-05, \
#            'l2_reg_embedding': 1e-05, 'l2_reg_cross': 1e-05, 'l2_reg_dnn': 0,\
#            'dnn_dropout': False, 'dnn_use_bn': False, 'dnn_activation': 'relu', 'task': 'binary'})}
#
#        self.modelsDict = {}
#        for modelName in models:
#            if modelName not in modelsDict.keys():
#                raise ValueError('Unsupported model {}'.format(modelName))
#                
##            query = '{}'.format(modelsDict[modelName])
#            model = modelsDict[modelName][0]
#            self.modelsDict[modelName] =  modelsDict[model]    
        
