import pandas as pd
import random
import numpy as np
np.random.seed(0)
random.seed(0)


df = pd.read_csv('train.csv',index_col='Id')
df.columns = df.columns.str.replace(" ", "")
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
validation = df[~msk]

train.to_csv('train_split.csv')
validation.to_csv('validation.csv')
