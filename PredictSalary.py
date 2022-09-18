# Izael Manuel Rascón Durán A01562240

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import BGDLinearRegressor as bgd


def rmse(y, y_hat):
    return np.sqrt(np.square(y_hat - y).mean())


# Load data
salary_data = pd.read_csv('Salary_Data.csv')
X = salary_data.YearsExperience.to_numpy()
y = salary_data.Salary.to_numpy()
X = X.reshape(X.shape[0], 1)
y = y.reshape(y.shape[0], 1)

bgd_regressor = bgd.BGDLinearRegressor()
bgd_regressor.fit(X, y, learning_rate=0.01, n_iterations=1000)

X_model = np.linspace(min(X), max(X), num=100)  # Data for plotting the regression line
prediction = bgd_regressor.predict(X_model)

# Plot the model
plt.plot(X_model, prediction, "r-")
plt.plot(X, y, "b.")
plt.title('Adjustment of the linear model')
plt.legend(['Adjusted linear model', 'Test data'])
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Calculate the root-mean-square error of the original data
salary_rmse = rmse(y, bgd_regressor.predict(X)).round(5)
print(f'The RMSE of the adjusted linear model is: {salary_rmse:.4e}')

