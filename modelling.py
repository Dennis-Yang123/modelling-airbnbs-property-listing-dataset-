# %%
from tabular_data import load_airbnb
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

training_data = load_airbnb("Price_Night")
# print(training_data)
np.random.seed(2)
x_train, x_test, y_train, y_test = train_test_split(training_data[0], training_data[1], test_size=0.3, random_state=42)

linear_model = SGDRegressor()
linear_model.fit(x_train, y_train)

y_test_prediction = linear_model.predict(x_test)
print(y_test_prediction)

y_mae = mean_absolute_error(y_test, y_test_prediction)
y_mse = mean_squared_error(y_test, y_test_prediction)
y_rmse = sqrt(y_mse)
y_r2 = r2_score(y_test, y_test_prediction)

print(f"MAE: {y_mae}")
print(f"MSE: {y_mse}")
print(f"RMSE: {y_rmse}")
print(f"R-Squared: {y_r2}")
# %%
