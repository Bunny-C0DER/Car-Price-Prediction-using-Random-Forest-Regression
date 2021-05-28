#importing libraires

import pandas as pd

#loading dataset

dataset = pd.read_csv('dataset.csv')
dataset = dataset.drop(['car_ID'],axis=1)

#summarizing dataset

print(dataset.shape)
print(dataset.head(5))

#splitting dataset into X and Y

Xdata = dataset.drop('price',axis='columns')
numericalCols=Xdata.select_dtypes(exclude=['object']).columns
X=Xdata[numericalCols]
X

Y = dataset['price']
Y

#scaling the Independent Variables (Features)


from sklearn.preprocessing import scale
cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
X

#splitting dataset into Train and Test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=0)

#traing using Random Forest

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train, y_train)

#Evaluating Model

ypred = model.predict(x_test)

from sklearn.metrics import r2_score
r2score = r2_score(y_test,ypred)
print("R2Score",r2score*100)