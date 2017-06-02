import pandas as pd
import numpy as np
import scipy.misc as ms
from sklearn.neural_network import MLPClassifier

os.chdir('/home/kishlay/Documents/Data Analytics Practice Problems/Identify The Digits')


#Read train file
df = pd.read_csv('train.csv')
print(df.head())
print(df.isnull().any())


#Read training images and flatten it to analyse
templist = []
for i in range(df.shape[0]):
    temp = ms.imread('./train/' + df.iloc[i].iloc[0],flatten=True)
    temp = temp.flatten()
    templist.append(temp)

X_train = np.stack(templist)
y_train = np.array(df['label'])
model = MLPClassifier(solver='lbfgs',alpha=1e-2,hidden_layer_sizes=(40,20),random_state=1)
model.fit(X_train, y_train)


#Read test data
df2 = pd.read_csv('test.csv')

#Read test data
templist = []
for i in range(df2.shape[0]):
    temp = ms.imread('./test/' + df2.iloc[i].iloc[0],flatten=True)
    temp = temp.flatten()
    templist.append(temp)
X_test = np.stack(templist)

results = model.predict(X_test)

df2['label'] = pd.Series(results)
df2.to_csv('./result.csv',index=False)