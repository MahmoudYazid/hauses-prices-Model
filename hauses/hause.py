import pandas as pd
import  matplotlib.pyplot as plt
import tensorflow.keras as k

xl=pd.read_csv('kc_house_data.csv')
x=xl[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']]
x_train=x.values
y=xl['price']
y_train=y.values

print(x_train.shape)
print(y_train.shape)
model=k.Sequential()
model.add(k.layers.Dense(512,input_dim=18,activation='relu'))
model.add(k.layers.Dropout(.1))
model.add(k.layers.Dense(512,activation='relu'))
model.add(k.layers.Dropout(.1))
model.add(k.layers.Dense(512,activation='relu'))
model.add(k.layers.Dropout(.25))
model.add(k.layers.Dense(1,activation='relu'))
model.compile(loss='msle',optimizer='RMSprop')
history=model.fit(x_train,y_train,epochs=30)

pre_sheet=pd.read_csv('testnew.csv')
target=pre_sheet[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']]
target_=target.values
import numpy as np
target_v=np.array(target_[3]).reshape((1,18))
predict=model.predict([target_v])
loss=history.history['loss'][int(len(history.history['loss'])-1)]
subfinal=float(predict)*loss/100
final=predict+subfinal
print(final)
plt.plot(history.history['loss'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
plt.show()
#id	date	price	bedrooms	bathrooms	sqft_living	sqft_lot	floors	waterfront	view	condition	grade	sqft_above	sqft_basement	yr_built	yr_renovated	zipcode	lat	long	sqft_living15	sqft_lot15
