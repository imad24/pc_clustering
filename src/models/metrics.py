import numpy as np


def getMSE(true,pred):
    return (( (true - pred) ** 2)).mean()
def getMAE(true,pred):
     return (np.abs((true - pred))).mean()

def getMAPE(true,pred):
    return np.abs((true - pred)/true).mean()

def getsMAPE(true,pred):
    return  ( (200*np.abs(true - pred)) / true+pred).mean()


def getMASE(training_series, testing_series, prediction_series,m=1):

    T = training_series.shape[0]
    diff=[]
    for i in range(m, T):
        value = training_series[i] - training_series[i - m]
        diff.append(value)
    d = np.abs(diff).sum()/(T-m)
    errors = np.abs(testing_series - prediction_series )

    return errors.mean()/d