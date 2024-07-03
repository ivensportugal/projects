import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("diamonds.csv")

#print(data.head(3))

data.fillna(data.mean(numeric_only=True), inplace=True)
data.drop_duplicates(inplace=True)

Q1 = data.quantile(0.25, numeric_only=True)
Q3 = data.quantile(0.75, numeric_only=True)

IQR = Q3 - Q1

#data = data[
#  ~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)
#]
#print(data < (Q1 - 1.5 * IQR))
#data=[data.any()]

#sns.pairplot(data,x_vars=['carat'],y_vars=['price'],height=12,kind='scatter')

#plt.xlabel('Carat')
#plt.ylabel('Price')
#plt.title('Diamond Price Prediction - Carat vs Price')

#plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#X=data['carat','price']
#y=data['price']

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7, random_state=100)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train_scaled,y_train)

y_pred = linear_regression_model.predict(X_test_scaled)

print("Mean Squared Error: ", mean_squared_error(y_test,y_pred))
print("R2 Score: ", r2_score(y_test,y_pred))
