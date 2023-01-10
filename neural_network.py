# %% 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from tabular_data import load_airbnb
import numpy as np
from sklearn.model_selection import train_test_split

class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self, features, label):
        super().__init__()
        self.features, self.label = load_airbnb("Price_Night")
    
    def __getitem__(self, index):
        return (torch.tensor(self.features.to_numpy()[index]), self.label[index])

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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


# %%