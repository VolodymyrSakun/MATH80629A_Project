
import numpy as np

def getClassWeights(y):
    neg, pos = np.bincount(y)
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))         
    weight_for_0 = (1 / neg)*(total)/2.0 
    weight_for_1 = (1 / pos)*(total)/2.0        
    class_weight = {0: weight_for_0, 1: weight_for_1}  
    return class_weight