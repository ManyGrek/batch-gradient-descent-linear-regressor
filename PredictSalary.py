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

bgd_regressor = bgd.BGDLinearRegressor()
bgd_regressor.fit(X, y, learning_rate=0.1, n_iterations=1)

X_new = np.linspace(min(X), max(X), num=10)  # Data for plotting the regression line
prediction = bgd_regressor.predict(X_new)

# Plot the model
plt.plot(X_new, prediction, "r.")
plt.plot(X, y, "b.")
#plt.axis([0, 2, 0, 15])
plt.title('Adjustment of the linear model')
#plt.legend(['Adjusted linear model', 'Test data'])
plt.show()

# Calculate the root-mean-square error of the original data
salary_rmse = rmse(y, bgd_regressor.predict(X)).round(5)
print(f'The RMSE of the adjusted linear model is: {salary_rmse:.4e}')
