# Run Recommender

import os
from recommender import Recommender as R
import recommender
import sys
from datetime import timedelta
import lib2
import time
import pandas as pd
import random
import numpy as np
from tensorflow.keras.metrics import AUC
from tensorflow.keras.metrics import BinaryAccuracy
from datetime import date



def stackRecommendations(dataItems, dataScores):    
#    dataItems = userItemSortedDf4
#    dataScores = userItemScoresDf4

    dataList = []
    for c in dataItems.columns:
        i1 = dataItems[[c]]
        i1.columns = ['sProdCode']
        s1 = dataScores[[c]]
        s1.columns = ['fScore']        
        j = i1.join(s1, how='inner')
        dataList.append(j)
        
    data = pd.concat(dataList, axis=0)
    data['sClientCode'] = data.index
    data.reset_index(drop=True, inplace=True)    
    data['fScore'] = data['fScore'].astype(float)   
    data = data[data['sProdCode'] != '']

    data.drop_duplicates(subset=['sClientCode', 'sProdCode'], keep='first', inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    data.sort_values(['sClientCode', 'fScore'], ascending=False, inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data

def getResponseColumn(data, mapUserToItemPosterior):

    response = []
    users = list(data['sClientCode'].values)
    items = list(data['sProdCode'].values)
    
    for user, item in zip(users, items):
        itemsPosterior = mapUserToItemPosterior.get(user, set())
        if item in itemsPosterior:
            response.append(1)
        else:
            response.append(0)

    return response

def getScoreItem(data, item):

#    MaPaie

    data1 = data[data['sProdCode'] == item].copy(deep=True)

    yTrue = data1['response'].values
    yProba = data1['fScore'].values
    
    tNp = np.arange(0.01, 1, 0.01)
    tNp = [round(x, 4) for x in tNp]
    f1List = []
    scores = []
    for t in tNp:
        yPredict = np.where(yProba > t, 1, 0)
        score = lib2.getScore(yTrue, yPredict, labelsDict={0: 'No', 1: 'Buy'})
        scores.append(score)
        if score.get('Labels') is None:
            f1List.append(0)
        else:
            f1List.append(score['Labels']['Buy']['F1'])
        
    idx = recommender.argmax(f1List)

    threshold = tNp[idx]
    yPredict = np.where(yProba > threshold, 1, 0)
    score = lib2.getScore(yTrue, yPredict, labelsDict={0: 'No', 1: 'Buy'})

    return threshold, score

###############################################################################

if __name__ == "__main__":
        
    workDir = os.path.dirname(os.path.realpath('__file__'))
#    dataDir = os.path.join(workDir, 'data')
#    resultsDir = os.path.join(workDir, 'results')
    METRICS = [BinaryAccuracy(name='accuracy'), AUC(name='auc')]
    

    daysToBuyPrior = 3650 # 30 to 150
    daysToBuyPosterior = 365
    reference = 'right'
    dLeftMargin = None
    dRightMargin = date(2020, 10, 23)
    margins = (dLeftMargin, dRightMargin)
    
    trainPosterior = False # if True, use items from posterior period for user-item matrix
    nRecommendations = 10
    nUsersRecommend = 1000 # 
    validFrac = 0.2
    embedding_dim = 10
    nEpochs = 1000
    chunkSize = 100
    sPredSegmentCode = '{}_{}_{}'.format(daysToBuyPrior, daysToBuyPosterior, reference)   
    iCieId = 46
    threshLow = 0.8
    threshHi = 0.98
    
    dPredict = date.today()
    
    start_time = time.time()

    licensesDf = lib2.loadObject(os.path.join(workDir, 'transactions.dat'))
    licensesDf.rename(columns={'userId': 'user', 'itemId': 'item', 'ds': 'date_t'}, inplace=True)         
    licensesDf.sort_values(['user', 'date_t', 'item'], inplace=True)
    licensesDf.reset_index(drop=True, inplace=True)
        
#    select only users that did purchases last year
    dLast = licensesDf['date_t'].max()
    print('Last transaction:', dLast)
    dLeftMargin = dLast - timedelta(days=int(365*5))    
    recent = licensesDf[licensesDf['date_t'] >= dLeftMargin]
    usersActive = sorted(list(recent['user'].unique()))
    users = usersActive # 6598
    
#    cut data
    licensesDf = licensesDf[licensesDf['user'].isin(users)]
    licensesDf.sort_values(['user', 'date_t', 'item'], inplace=True)
    licensesDf.reset_index(drop=True, inplace=True)
    
#    users = list(licensesDf['user'].unique())
    itemsPredictList = sorted(list(licensesDf['item'].unique()))
           
    mapMissingToDefault = {'sparse': '_', 'dense': 0, 'varlen': ''}

    mapUserFeatureToType = {'feature_1': 'sparse', 'feature_2': 'sparse', 'feature_3': 'sparse',
        'feature_4': 'sparse', 'feature_5': 'sparse', 'feature_6': 'sparse',
        'feature_7': 'sparse', 'feature_8': 'sparse', 'feature_9': 'sparse',
        'feature_10': 'sparse', 'feature_11': 'sparse', 'feature_12': 'sparse', 'user': 'sparse'}   
    
    mapItemFeatureToType = {'item': 'sparse'}
    mapItemFeatureToDefault = {'itemOrig': '_', 'item': 0}
    
#   load users features
    userFeaturesDf = lib2.loadObject(os.path.join(workDir, 'userFeatures.dat'))
    userFeaturesDf.drop_duplicates(subset=['userId'], inplace=True, keep='last')
    userFeaturesDf.set_index('userId', drop=True, inplace=True)
    userFeaturesDf.sort_index(inplace=True)

    usersRecommend = random.sample(users, nUsersRecommend)
    itemsRecommend = sorted(itemsPredictList)
    itemsRecommend = [str(x) for x in itemsRecommend]
    
###############################################################################    
    
    
    r = R(usersTrain='all', itemsTrain='all', usersRecommend=usersRecommend, \
          itemsRecommend=itemsRecommend, trainPosterior=trainPosterior, \
          daysToBuyPrior=daysToBuyPrior, \
          daysToBuyPosterior=daysToBuyPosterior, reference=reference, \
          margins=margins, resultsDir=workDir)
                
    r.loadTransactions(licensesDf)

    r.loadUserFeatures(userFeaturesDf, mapUserFeatureToType,
        {}, mapMissingToDefault)

    itemFeaturesDf = pd.DataFrame(index=r.items)
    r.loadItemFeatures(itemFeaturesDf, mapItemFeatureToType, 
        {}, mapMissingToDefault)
    
#    1. User-user recommender based on history of purchases 
    userItemSortedDf1, userItemScoresDf1 = r.userUserRecommend1( \
        nRecommendations=None, rescale='all')   
    
#    2. Item-item recommender based on history of purchases
    userItemSortedDf2, userItemScoresDf2 = r.itemItemRecommend( \
        nRecommendations=None, rescale='all')

#   3. User-user based recommender based on cosine similarity of users features     
    userItemSortedDf3, userItemScoresDf3 = r.userUserRecommend2( \
        nRecommendations=None, rescale='all')
    
#   4. DeepCTR models
    r.makeUserItemTrain()
    r.makeUserItemRecommend()

#    modelName = 'WDL' # ['WDL', 'DeepFM', 'NFM']
    
    userItemSortedDf4, userItemScoresDf4 = r.annRecommend('WDL', embedding_dim=embedding_dim, 
        dropout=0.5, nEpochs=nEpochs, patience=10, batch_size=1024, validFrac=validFrac, 
        plot=True, getScore=True)

    scoreWDL = r.score
    thresholdWDL = r.thresholdBest
    scoreBestWDL = r.scoreBest
    
    userItemSortedDf5, userItemScoresDf5 = r.annRecommend('DeepFM', embedding_dim=embedding_dim, 
        dropout=0.5, nEpochs=nEpochs, patience=10, batch_size=1024, validFrac=validFrac, 
        plot=True, getScore=True)

    scoreDeepFM = r.score
    thresholdDeepFM = r.thresholdBest
    scoreBestDeepFM = r.scoreBest

    userItemSortedDf6, userItemScoresDf6 = r.annRecommend('NFM', embedding_dim=embedding_dim, 
        dropout=0.5, nEpochs=nEpochs, patience=10, batch_size=1024, validFrac=validFrac, 
        plot=True, getScore=True)

    scoreNFM = r.score
    thresholdNFM = r.thresholdBest
    scoreBestNFM = r.scoreBest
    
    end_time = time.time()
    t = end_time - start_time
    d, h, m, s = lib2.formatTime(t)
    print(d, 'days', h, 'hours', m, 'min', s, 'sec')


    mapUserToPredict1 = recommender.recommendationsToDict(userItemSortedDf1, userItemScoresDf1)
    mapUserToPredict2 = recommender.recommendationsToDict(userItemSortedDf2, userItemScoresDf2)
    mapUserToPredict3 = recommender.recommendationsToDict(userItemSortedDf3, userItemScoresDf3)
    mapUserToPredict4 = recommender.recommendationsToDict(userItemSortedDf4, userItemScoresDf4)
    mapUserToPredict5 = recommender.recommendationsToDict(userItemSortedDf5, userItemScoresDf5)
    mapUserToPredict6 = recommender.recommendationsToDict(userItemSortedDf6, userItemScoresDf6)

    recommendationsDictList = [mapUserToPredict1, mapUserToPredict2,\
        mapUserToPredict3, mapUserToPredict4, mapUserToPredict5, mapUserToPredict6]

#    recommendationsDictList = [mapUserToPredict4]

    cmDf = pd.DataFrame([['tn', 'fn'],['fp', 'tp']], index=['Predicted_0', 'Predicted_1'],\
        columns=['Actual_0', 'Actual_1'], dtype=int)

    orig_stdout = sys.stdout
    f = open(os.path.join(r.resultsDir, '{}.txt'.format(sPredSegmentCode)), 'w')
    sys.stdout = f

    print('Days to buy prior items: {}'.format(daysToBuyPrior))
    print('Days to buy posterior items: {}'.format(daysToBuyPosterior))
    print('Max number of recommendations: {}'.format(nRecommendations))
    print('Validation fraction: {}'.format(validFrac))
    print('embedding_dim: {}'.format(embedding_dim))
    
    print('Confusion matrix: \n', cmDf, '\n')

    print('WDL normal threshold == 0.5:\n', scoreWDL, '\n')
    print('WDL threshold == {}:\n'.format(thresholdWDL), scoreBestWDL, '\n')

    print('DeepFM normal threshold == 0.5:\n', scoreDeepFM, '\n')
    print('DeepFM threshold == {}:\n'.format(thresholdDeepFM), scoreBestDeepFM, '\n')

    print('NFM normal threshold == 0.5:\n', scoreNFM, '\n')
    print('NFM threshold == {}:\n'.format(thresholdNFM), scoreBestNFM, '\n')
    
    print("""
    nIntersection = len(trueItemsPosterior.intersection(topRecomItems))
    nUnion = len(trueItemsPosterior.union(topRecomItems))      
    tp = len(topRecomItems.intersection(trueItemsPosterior))
    fp = nRecommendations - tp
    tmp1 = itemsSet.difference(trueItemsPosterior)
    tmp2 = itemsSet.difference(topRecomItems)
    tn = len(tmp1.intersection(tmp2))
    fn = mPosterior - tp
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Jaccard = nIntersections / nUnion for first N recommendationd
    Precision = tp / (tp + fp) or tp / nRecommendations
    Recall = tp / (tp + fn)
    F1 = 2 * Precision * Recall / (Precision + Recall)    
    """)    
    

    nList = [1, 2, 3, 4, 5]

    for n in nList:   
#   n = 1
        print('Number of first top recommendations: {}\n'.format(n))

        for algorithm, d in enumerate(recommendationsDictList):

            print('Algorithm {}'.format(algorithm+1))              
                        
            scores1 = recommender.getScoresCF(d, r.mapUserToItemPosterior, itemsRecommend, n=n)
            
            accuracy = scores1['accuracy'][n]
            print('Accuracy for first {} recommendations: {:7.6f}'.format(n, np.mean(accuracy)))
            
            jaccard = scores1['jaccard'][n]
            print('Jaccard for first {} recommendations: {:7.6f}'.format(n, np.mean(jaccard)))
            
            precision = scores1['precision'][n]        
            print('Precision for first {} recommendations: {:7.6f}'.format(n, np.mean(precision)))
            
            recall = scores1['recall'][n]        
            print('Recall for first {} recommendations: {:7.6f}'.format(n, np.mean(recall)))
            
            f1 = scores1['f1'][n]                  
            print('F1 for first {} recommendations: {:7.6f}'.format(n, np.mean(f1)))
    
            print()
        
        print('Number of test clients that have {} posterior items: {}\n\n'.format(n, len(f1)))
                    
    sys.stdout = orig_stdout
    f.close()

    dataPredRecent = recommender.makePredictions(recommendationsDictList, r.mapUserToItem, \
        r.mapUserToItemPrior, {}, nRecommendations=nRecommendations)

    dataPredRecent.to_excel(os.path.join(workDir, '{}. {} recommendations. {} users.xlsx'.\
        format(sPredSegmentCode, nRecommendations, len(r.usersRecommend))), index=False, na_rep='NaN')






