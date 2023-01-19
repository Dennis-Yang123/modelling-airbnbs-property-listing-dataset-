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

class LinearRegression(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = torch.nn.Sequential()
        self.layers.add_module("input_layer", torch.nn.Linear(9, config["hidden_layer_width"]))
        self.layers.add_module("activation_layer", torch.nn.ReLU())
        for i in range(config["model_depth"] - 1):
            self.layers.add_module(f"hidden_layer_width", torch.nn.Linear(config["hidden_layer_width"], config["hidden_layer_width"]))
            self.layers.add_module("output_layer", torch.nn.Linear(config["hidden_layer_width"], 1))
    
    def forward(self, features):
        return self.layers(features)

def train(model, dataloader, epoch, config):
    optimiser_name = config["optimiser"]
    if optimiser_name == "SGD":
        optimiser = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    elif optimiser_name == "Adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    else:
        raise ValueError(f"Optimiser: {optimiser_name} not supported.")
    batch_index = 0
    writer = SummaryWriter()
    for epoch in range(epoch):
        for batch in dataloader:
            features, labels = batch
            prediction = model(features)
            labels = labels.to(prediction.dtype)
            loss = F.mse_loss(prediction, labels)
            loss.backward()
            print(loss.item())
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar("loss", loss.item(), batch_index)
            batch_index += 1

def get_nn_config():
    with open(r"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\nn_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    return config
        
if __name__ == "__main__":
    config = get_nn_config()
    model = LinearRegression(config)
    train(model, train_loader, 25, config)
    # print(dict_hyper)
# %%