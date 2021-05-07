import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('sampleData.csv')
data.head()

data.shape

print(data.isnull().sum())


data= data.dropna()

#print(data.tail())

print(data.isnull().sum())


from sklearn.preprocessing import StandardScaler
data['NormalisAmount']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data.head()

data.drop(['Amount','Time'],axis=1,inplace=True) 

#print(data.head())

shuffl_df = data.sample(frac=1,random_state=4)
shuffl_df.tail()

fraud=shuffl_df.loc[shuffl_df['TARGET']==1]
non_fraud=shuffl_df.loc[shuffl_df['TARGET']==0].sample(n=450,random_state=43)

data=pd.concat([fraud,non_fraud])
data = data.sample(frac=1,random_state=4)


from sklearn.model_selection import train_test_split

x=data.iloc[:,data.columns!='TARGET']
y=data.iloc[:,data.columns=='TARGET']

xtrain, xtest, ytrain, ytest =train_test_split(x,y, test_size=0.3)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units =15 , kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))
# Adding the second hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(xtrain, ytrain, batch_size = 32, epochs = 100)


# Predicting the Test set results
y_pred = classifier.predict(xtest)
y_pred = (y_pred > 0.5)
score = classifier.evaluate(xtest, ytest)
print(score)

from sklearn.metrics import confusion_matrix

cnf_model=confusion_matrix(ytest,y_pred)
print(cnf_model)  #TN #FP #FN #TP

#finding correlation between columns and plotting heatmap

corrmat = data.corr() 
fig = plt.figure(figsize = (12, 9)) 
sns.heatmap(corrmat, vmax = .8, square = True) 
plt.show()


LABELS = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(ytest,y_pred) 
plt.figure(figsize =(12, 12)) 
sns.heatmap(conf_matrix, xticklabels = LABELS,  
            yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show()

"""
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

acc1=accuracy_score(ytest,y_pred)
print(' accuracy score:',acc1)


a = float(input("REGION_RATING_CLIENT_W_CITY : "))
b= float(input("REG_CITY_NOT_WORK_CITY "))
c = float(input("REG_REGION_NOT_LIVE_REGION: "))
d= float(input("REG_REGION_NOT_WORK_REGION: "))
e= float(input("LIVE_REGION_NOT_WORK_REGION: "))
f= float(input("REG_CITY_NOT_LIVE_CITY: "))
h= float(input("LIVE_CITY_NOT_WORK_CITY: "))
i= float(input("Time: "))
j= float(input("Amount: "))

result = classifier.predict([[a,b,c,d,e,f,h,i,j]])  # input must be 2D array

print(result)
"""