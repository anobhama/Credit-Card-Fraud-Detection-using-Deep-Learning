import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import sklearn.metrics

df = pd.read_csv('sampleData.csv', low_memory=False)
X = df.iloc[:,:-1]
y = df['TARGET']
df.head()
frauds = df.loc[df['TARGET'] == 1]
non_frauds = df.loc[df['TARGET'] == 0]
print("We have", len(frauds), "fraud data points and", len(non_frauds), "regular data points.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print("Size of training set: ", X_train.shape)
model = Sequential()
model.add(Dense(30, input_dim=30, activation='relu'))     # kernel_initializer='normal'
model.add(Dense(1, activation='sigmoid'))                 # kernel_initializer='normal'
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
"""
model.fit(X_train.values(),y_train,epochs=1)
print("Loss: ", model.evaluate(X_test.values(), y_test, verbose=0))
y_predicted = model.predict(X_test.values()).T[0].astype(int)
from sklearn.metrics import confusion_matrix
y_right = np.array(y_test)
confusionmatrix = confusion_matrix(y_right, y_predicted)
print("Confusion matrix:\n%s" % confusionmatrix)
confusionmatrix.plot(normalized=True)
plt.show()
confusionmatrix.print_stats()

from sklearn.decomposition import *
from sklearn.preprocessing import *

df2 = pd.ModelFrame(X_train, target=y_train)
sampler = df2.imbalance.over_sampling.SMOTE()
oversampled = df2.fit_sample(sampler)
X2, y2 = oversampled.iloc[:,:-1], oversampled['TARGET']

data = scale(X2)
pca = PCA(n_components=10)
X2 = pca.fit_transform(data)
X2model2 = Sequential()
X2model2.add(Dense(10, input_dim=10, activation='relu'))
X2model2.add(Dense(27, activation='relu'))
X2model2.add(Dense(20, activation='relu'))
X2model2.add(Dense(15, activation='relu'))
X2model2.add(Dense(1, activation='sigmoid'))
X2model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
X2model2.summary()
X2_test = pca.fit_transform(X_test)
h = X2model2.fit(X2, y2, epochs=5, validation_data=(X2_test, y_test))
print("Loss: ", X2model2.evaluate(X2_test, y_test, verbose=2))
y2_predicted = np.round(X2model2.predict(X2_test)).T[0]
y2_correct = np.array(y_test)
confusionmatrix2 =confusion_matrix(y2_predicted,y2_correct)
print("Confusion matrix:\n%s" % confusionmatrix2)
confusionmatrix2.plot(normalized=True)
plt.show()
confusionmatrix2.print_stats()
"""