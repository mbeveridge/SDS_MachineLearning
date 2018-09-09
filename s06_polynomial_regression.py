# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/s6_Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set [NB. `sklearn.cross_validation` is deprecated]
# Not done in this case, because only ten observations are available
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset [§6 Lect57: "Polynomial Regression - Python pt2" ...for comparison]
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset [§6 Lect57: "Polynomial Regression - Python pt2"]
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # Video eventually says (~15'00-17'10) to use degree = 4, not degree = 2
X_poly = poly_reg.fit_transform(X)
# cont'd : from 9min20 : include this fit (that we made with our `poly_reg` object 
# and our `X_poly` matrix of polynomial features) into a MLR model
# poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results [§6 Lect58: "Polynomial Regression - Python pt3" ...for comparison]
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results [§6 Lect58: "Polynomial Regression - Python pt3"]
plt.scatter(X, y, color = 'red')
# plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (add X_grid ...if we want higher resolution and smoother curve)
# [§6 Lect58: "Polynomial Regression - Python pt3" from 18min00]
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1)) # to turn a vector into a matrix
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression [§6 Lect59: "Polynomial Regression - Python pt4" ...for comparison]
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression [§6 Lect59: "Polynomial Regression - Python pt4"]
lin_reg_2.predict(poly_reg.fit_transform(6.5))