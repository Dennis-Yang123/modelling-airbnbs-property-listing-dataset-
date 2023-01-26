# %%
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from tabular_data import load_airbnb
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import yaml
from modelling import save_model
from sklearn.metrics import r2_score
import time
from datetime import datetime
import random
import os

class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self, features, label):
        super().__init__()
        self.features, self.label = load_airbnb("Price_Night")
    
    def __getitem__(self, index):
        return (torch.from_numpy(self.features.to_numpy()[index]).float(), self.label[index])

    def __len__(self):
        return len(self.label)

data = load_airbnb("Price_Night")
# print(data[1])
# print(len(data))
x_train, x_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.2, random_state=42)

# Further split the train set into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

train_dataset = AirbnbNightlyPriceImageDataset(x_train, y_train)
test_dataset = AirbnbNightlyPriceImageDataset(x_test, y_test)
val_dataset = AirbnbNightlyPriceImageDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=4   , shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

# class LinearRegression(torch.nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.layers = torch.nn.Sequential(
#             torch.nn.Conv1d(4, 4, 1),
#             torch.nn.ReLU(),
#             torch.nn.Conv1d(4, 9, 1),
#             torch.nn.ReLU(),
#             torch.nn.Flatten()
#             )
#         for i in range(config["model_depth"] - 1):
#             self.layers.add_module("linear_layer", torch.nn.Linear(9, config["hidden_layer_width"]))
#             self.layers.add_module("relu", torch.nn.ReLU())   

#         self.layers.add_module("output_layer", torch.nn.Linear(config["hidden_layer_width"], 1))
#         self.layers.add_module("Relu", torch.nn.ReLU())

#     def forward(self, features):
#         print(features.shape)
#         print(self.layers(features))
#         return self.layers(features)

class LinearRegression(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = torch.nn.Sequential()
        self.layers.add_module("input_layer", torch.nn.Linear(9, 9))
        self.layers.add_module("activation_layer", torch.nn.ReLU())
        for i in range(config["model_depth"] - 2):
            self.layers.add_module(f"hidden_layer_width", torch.nn.Linear(9, config["hidden_layer_width"]))
            self.layers.add_module("Relu", torch.nn.ReLU())
        self.layers.add_module("output_layer", torch.nn.Linear(config["hidden_layer_width"], 1))
    
    def forward(self, features):
        return self.layers(features)


def train(model, dataloader, epoch, config):
    start_time = time.time()
    dt_now = datetime.now()
    optimiser_name = config["optimiser"]
    if optimiser_name == "SGD":
        optimiser = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    elif optimiser_name == "Adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif optimiser_name == "Adagrad":
        optimiser = torch.optim.Adagrad(model.parameters(), lr=config["learning_rate"])
    elif optimiser_name == "Adadelta":
        optimiser = torch.optim.Adadelta(model.parameters(), lr=config["learning_rate"])
    else:
        raise ValueError(f"Optimiser: {optimiser_name} not supported.")
    batch_index = 0
    writer = SummaryWriter()
    prediction_list = []
    labels_list = []
    num_predictions = 0
    for epoch in range(epoch):
        for batch in dataloader:
            features, labels = batch
            prediction = model(features)
            prediction_list.append(prediction)
            labels_list.append(labels.detach().numpy())
            labels = labels.to(prediction.dtype)
            loss = F.mse_loss(prediction, labels)
            loss.backward()
            rmse_loss = torch.sqrt(loss)
            print(loss.item())
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar("loss", loss.item(), batch_index)
            batch_index += 1

    num_predictions += prediction.shape[0]

    labels = np.concatenate(labels_list)
    prediction_list = np.concatenate([pred.detach().numpy() for pred in prediction_list])    
    r2 = r2_score(labels, prediction_list)

    end_time = time.time()
    total_time = end_time - start_time
    dt_string = dt_now.strftime("%d_%m_%Y_%H-%M")
    inference_latency = total_time / num_predictions
    
    best_metrics = {
        "RMSE_loss": str(rmse_loss), 
        "R_squared": r2, 
        "training_duration": total_time,
        "inference_latency": inference_latency
    }
    
    print(best_metrics)
    return best_metrics, dt_string


def get_nn_config():
    with open(r"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\nn_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    return config

def generate_nn_configs():
    optimiser = ["SGD", "Adam", "Adagrad", "Adadelta"]
    learning_rate = [0.01, 0.001, 0.0001]
    hidden_layer_width = [3, 4, 5, 6, 7, 8, 9]
    model_depth = [3, 4, 5, 6]
    config_list = []

    for index in range(0,17):
        
        config_file = {
        "optimiser": random.choice(optimiser),
        "learning_rate": random.choice(learning_rate),
        "hidden_layer_width": random.choice(hidden_layer_width),
        "model_depth": random.choice(model_depth)
    }
        config_list.append(config_file)
    print(config_list)
    return config_list

def find_best_nn(config_list):
    # best_metric_list = []
    best_r2 = -float("inf")
    best_model = None
    best_config = None
    best_metrics_dict = None
    for config in config_list:
        model = LinearRegression(config)
        best_metrics, dt_string = train(model, train_loader, 25, config)    

        if best_metrics["R_squared"] > best_r2:
            best_r2 = best_metrics["R_squared"]
            best_model = model
            best_config = config
            best_metrics_dict = best_metrics

    os.mkdir(f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\regression\\neural_networks\{dt_string}")
    torch.save(best_model.state_dict, f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\regression\\neural_networks\\{dt_string}\model.pt")
    return best_config, best_metrics_dict, dt_string, best_model

if __name__ == "__main__":
    # config = get_nn_config()
    # model = LinearRegression(config)
    # best_metrics, dt_string = train(model, train_loader, 25, config)
    
    config_list = generate_nn_configs()
    best_hyperparameters, best_metrics, dt_string, best_model = find_best_nn(config_list)
    save_model("neural_networks", best_model, best_hyperparameters, best_metrics, dt_string)
# %%