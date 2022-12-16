# %%
from tabular_data import load_airbnb
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
import numpy as np

training_data = load_airbnb("Price_Night")
# print(training_data)
np.random.seed(2)
x_train, x_test, y_train, y_test = train_test_split(training_data[0], training_data[1], test_size=0.3, random_state=42)

linear_model = SGDRegressor()
linear_model.fit(x_train, y_train)

y_test_prediction = linear_model.predict(x_test)


print(y_test_prediction)
# %%
