import pandas as pd
import random
import numpy as np
np.random.seed(0)
random.seed(0)


train = pd.read_csv('data/train.csv',index_col='Id').drop(columns="BUTTER")
train.columns = train.columns.str.replace(" ", "")

try:
    train = train.drop(columns='signal')
except:
    pass

test = pd.read_csv('data/test.csv',index_col='Id').drop(columns="BUTTER")
test.columns = test.columns.str.replace(" ", "")

try:
    test= test.drop(columns='signal')
except:
    pass



full = train.append(test)

full.to_csv('data/full.csv')