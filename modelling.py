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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

training_data = load_airbnb("Price_Night")
# print(training_data)
np.random.seed(2)
x_train, x_test, y_train, y_test = train_test_split(training_data[0], training_data[1], test_size=0.3, random_state=42)

linear_model = SGDRegressor()
linear_model.fit(x_train, y_train)

# y_test_prediction = linear_model.predict(x_test)
# print(y_test_prediction)

# y_mae = mean_absolute_error(y_test, y_test_prediction)
# y_mse = mean_squared_error(y_test, y_test_prediction)
# y_rmse = sqrt(y_mse)
# y_r2 = r2_score(y_test, y_test_prediction)

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


def tune_regression_model_hyperparameters(untuned_model, features, labels, dict_hyper):
    grid_search = GridSearchCV(untuned_model, dict_hyper, cv = 5)
    grid_search.fit(features, labels)
    # print(grid_search.best_params_)
    best_parameters = grid_search.best_params_
    best_rmse = sqrt(abs(grid_search.best_score_))

    return best_parameters, best_rmse


def save_model(folder, model, best_parameters, best_rmse):
    try:
        os.mkdir(f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\regression\{folder}")
        joblib.dump(model, f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\regression\\{folder}\model.joblib")

        with open(f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\regression\\{folder}\hyperparameters.json", "w") as f:
            json.dump(best_parameters, f)
        
        with open(f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\regression\\{folder}\metrics.json", "w") as f:
            json.dump(best_rmse, f)

    except FileExistsError:
        print("Folder or file already exists, will overwrite with new data")
        joblib.dump(model , f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\regression\\{folder}\model.joblib")

        with open(f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\regression\\{folder}\hyperparameters.json", "w") as f:
            json.dump(best_parameters, f)
        
        with open(f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\regression\\{folder}\metrics.json", "w") as f:
            json.dump(best_rmse, f)

def evaluate_all_models(model, dict_hyper):
    best_parameters, best_rmse = tune_regression_model_hyperparameters(model, training_data[0], training_data[1], dict_hyper)
    save_model(str(model), model, best_parameters, best_rmse)
    print(best_parameters)
    print(best_rmse)

    return best_parameters, best_rmse



if __name__ == "__main__":  
    dict_hyper = {
    'learning_rate': [0.1, 0.2, 0.3],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5]
    } #  Change this dictionary to the relevant model hyperparameters       
    
    evaluate_all_models(GradientBoostingRegressor(), dict_hyper) # Change argument for what model you desire
    

# %%
