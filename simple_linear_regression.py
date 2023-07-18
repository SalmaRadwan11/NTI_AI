# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#  conda install pandas 


# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values   #input < matrix 2D [[1.1][2][1.5]]>[1.1,2,1.5]
y = dataset.iloc[:, 1].values    #output < vector 1D

print(type(X))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3.0, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set

#   data > algorithm >> learning >> model >> test


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  # object from the class
regressor.fit(X_train, y_train)  # learning 
b=regressor.intercept_
A=regressor.coef_
# model <<
# Predicting the Test set results
y_pred = regressor.predict(X_test)


# Y=A.X+b


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#plt.plot(X_train, y_train, color = 'blue')

plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()











from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_pred)

def mm(x,y):
    dif=0
    for i in range(len(x)):
        dif=dif+(x[i]-y[i])**2
    return dif/len(x)
mm(y_test,y_pred)


regressor.predict([[8.8]])



