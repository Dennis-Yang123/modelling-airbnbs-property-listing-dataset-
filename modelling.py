# %%
from tabular_data import load_airbnb
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from sklearn.model_selection import GridSearchCV
import os
import json
import joblib

training_data = load_airbnb("Price_Night")
# print(training_data)
np.random.seed(2)
x_train, x_test, y_train, y_test = train_test_split(training_data[0], training_data[1], test_size=0.3, random_state=42)

linear_model = SGDRegressor()
linear_model.fit(x_train, y_train)

y_test_prediction = linear_model.predict(x_test)
# print(y_test_prediction)

y_mae = mean_absolute_error(y_test, y_test_prediction)
y_mse = mean_squared_error(y_test, y_test_prediction)
y_rmse = sqrt(y_mse)
y_r2 = r2_score(y_test, y_test_prediction)

# print(f"MAE: {y_mae}")
# print(f"MSE: {y_mse}")
# print(f"RMSE: {y_rmse}")
# print(f"R-Squared: {y_r2}")

def custome_tune_regression_model_hyperparameters(model, features, label, dict_hyp):
    best_params = None
    best_score = float("inf")

    for params in dict_hyp:
        model = SGDRegressor(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        score = sqrt(mean_squared_error(y_test, y_pred))

        if score < best_score:
            best_params = params
            best_score = score
            dict_metric = {"validation_RMSE": score}

    return best_params, dict_metric
dict_hyp = {
    'loss': ['huber'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': [0.1, 0.01, 0.001],
}

# best_parameters, best_score= custome_tune_regression_model_hyperparameters(linear_model, training_data[0], training_data[1], dict_hyp)
# print(best_parameters)
# print(best_score)

def tune_regression_model_hyperparameters():
    grid_search = GridSearchCV(linear_model, dict_hyp, cv = 5)
    grid_search.fit(training_data[0], training_data[1])
    # print(grid_search.best_params_)
    best_parameters = grid_search.best_params_
    best_rmse = sqrt(-grid_search.best_score_)

    return best_parameters, best_rmse

best_parameters, best_rmse = tune_regression_model_hyperparameters()
tuned_model = SGDRegressor(best_parameters)

def save_model(folder):
    try:
        os.mkdir(f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\regression\{folder}")
        joblib.dump(tuned_model, f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\regression\\{folder}\model.joblib")

        with open(f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\regression\\{folder}\hyperparameters.json", "w") as f:
            json.dump(best_parameters, f)
        
        with open(f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\regression\\{folder}\metrics.json", "w") as f:
            json.dump(best_rmse, f)

    except FileExistsError:
        print("Folder or file already exists, will overwrite with new data")
        joblib.dump(tuned_model, f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\regression\\{folder}\model.joblib")

        with open(f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\regression\\{folder}\hyperparameters.json", "w") as f:
            json.dump(best_parameters, f)
        
        with open(f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\regression\\{folder}\metrics.json", "w") as f:
            json.dump(best_rmse, f)
            
save_model("linear_regression")
       


# %%
