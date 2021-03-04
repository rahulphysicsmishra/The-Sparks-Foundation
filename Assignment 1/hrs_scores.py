import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Importing the data and printing first 5 rows
score_hour_data = pd.read_csv('hours_scores.csv')
print(score_hour_data.head())

# Separating input and output
X = np.c_[score_hour_data['Hours']]
y = np.c_[score_hour_data['Scores']]

# Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Plotting the data to have an idea
score_hour_data.plot(kind='scatter', x='Hours', y='Scores')
plt.show()

# Fitting the model for linear regression
model = LinearRegression()
model.fit(X_train, y_train)

pred_TSF_score = model.predict([[9.5]])
print(pred_TSF_score)  # It will come as 94.806 Score

# predicting the model on X_train dataset for mean squared error
pred_train_data = model.predict(X_train)


# checking  root mean squared error
from sklearn.metrics import mean_squared_error
mse_train = mean_squared_error(y_train, pred_train_data)
rmse_train_data = np.sqrt(mse_train)  # It will come 5.608
print(rmse_train_data)

# checking rmse on test data
pred_test_data = model.predict(X_test)
mse_test = mean_squared_error(y_test, pred_test_data)
rmse_test_data = np.sqrt(mse_test)  # It will come 4.352
print(rmse_test_data)


