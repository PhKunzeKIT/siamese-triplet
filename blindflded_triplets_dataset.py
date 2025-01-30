import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader


class BlindfoldedTripletsDataset(datasets.VisionDataset):
    def __init__(self, filepath, sensor_keys, train=True, test_size=0.2, n_selection=None, preloaded_data=None) -> None:
        self.filepath = filepath
        self.sorted_sensor_keys = sorted(sensor_keys)
        self.n_selection = n_selection
        self.train = train
        self.places = self.load_places_from_json()
        self.data = self.compile_data()
        self.labels = self.compile_labels()
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            self.data, self.labels, test_size=test_size, random_state=42)
        
    def __len__(self) -> int:
        if self.train:
            len = self.train_data.shape[0]
        else:
            len = self.test_data.shape[0]
        return len

    def __getitem__(self, idx) -> tuple:
        if self.train:
            data = self.train_data[idx]
            label = int(self.train_labels[idx])
        else:
            data = self.test_data[idx]
            label = int(self.test_labels[idx])
        data = data.unsqueeze(0) # Ensure the data tensor has shape [1, m, n]
        return data, label
    
    def compile_data(self) -> torch.tensor:
        """
        compiles all sensor data from the places into a single tensor
        """
        compiled_data = torch.empty((0, len(self.sorted_sensor_keys), 2400))
        for place in self.places:
            for crossing_key in place['IMU_windows_interpolated'].keys():
                imu_data = self.convert_IMU_dict_to_tensor(place['IMU_windows_interpolated'][crossing_key])
                compiled_data = torch.cat((compiled_data, imu_data.unsqueeze(0)), dim=0)
        return compiled_data
    
    def compile_labels(self) -> torch.tensor:
        """
        compiles all labels from the places into a single tensor
        """
        compiled_labels = torch.empty(0, dtype=torch.long)
        for place in self.places:
            for crossing_key in place['IMU_windows_interpolated'].keys():
                label = torch.tensor(place['id'])
                compiled_labels = torch.cat((compiled_labels, label.unsqueeze(0)), dim=0)
        return compiled_labels

    def load_places_from_json(self, preloaded_data=None) -> dict:
        """
        loads places from a .json file and returns
        them in dictionary format
        """
        with open(self.filepath) as f:
            if preloaded_data == None:
                print("load_places_from_json(): loading .json file")
                data = json.load(f)
                print(f'load_places_from_json(): loaded {len(data.keys())} items')
            else:
                print("load_places_from_json(): using testing data")
                data = preloaded_data
            places = dict()
            if self.n_selection == None:
                data_items = list(data.items())
            else:
                data_items = list(data.items())[:self.n_selection]
            for key, value in data_items:
                places[key] = value
            del data
        return list(places.values())
    
    def convert_IMU_dict_to_tensor(self, data_dict) -> torch.tensor:
        """
        creates a list of individual torch.tensors of all entries in the data dict
        and stacks them to a single 2D tensor as output
        """
        tensors = [torch.tensor(data_dict[key]) for key in self.sorted_sensor_keys]
        data_tensor = torch.stack(tensors)
        del tensors
        return data_tensor
    
    def tensor_index_translation(self, sensor_key) -> int:
        """
        translates the original sensor key to the new index in the tensor for easier access
        """
        translation_dict = {key: idx for idx, key in enumerate(self.sorted_sensor_keys)}
        return translation_dict[sensor_key]


if __name__ == '__main__':
    pass