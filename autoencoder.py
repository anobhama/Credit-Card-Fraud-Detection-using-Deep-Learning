import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

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

#autoencoders
input_dim = xtrain.shape[1]
encoding_dim = 9
input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh", 
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

#building the model

nb_epoch = 100
batch_size = 32
autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',histogram_freq=0,write_graph=True,write_images=True)
history = autoencoder.fit(xtrain, xtrain,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(xtest, xtest),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history

"""
autoencoder = load_model('model.h5')

predictions = autoencoder.predict(xtest)
mse = np.mean(np.power(xtest - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,'true_class': ytest})

print(error_df.describe())
"""

y_pred = autoencoder.predict(xtest)
y_pred = (y_pred > 0.5)
score = autoencoder.evaluate(xtest, ytest)
print(score)
    
