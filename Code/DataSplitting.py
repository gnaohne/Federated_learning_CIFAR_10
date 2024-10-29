import os
import json
import torch
from torchvision import datasets, transforms
from typing import Dict, List
import random
from PIL import Image

class DataSplitting:
    def __init__(self, data_dir: str, data_split_file: str):
        self.data_dir = data_dir
        self.data_split_file = data_split_file

        self.train_data = None
        self.clients = None

        self._load_data()
        self._load_data_split()

    def _load_data(self):
        self.train_data = datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        # images_data, labels_data, self.labels = self.train_data.data, self.train_data.targets, self.train_data.classes
        
        # print(f"Training data size: {len(self.train_data)}")
        # print(f"Image size: {self.train_data.data.shape}")
        # print(f"Labels: {self.train_data.classes}")

    def _load_data_split(self):
        # Load data split configuration from JSON file
        with open(self.data_split_file, 'r') as f:
            self.data_split = json.load(f)

        print(f"Data split configuration:")
        for idx, val in enumerate(self.data_split['clients']):
            name, data = val['name'], val['data']
            print(f"Client {idx + 1}: {name} - {data}")

        self.clients_name = [val['name'] for idx, val in enumerate(self.data_split['clients'])]

        label_clientname_list = []
        for label in range(len(self.train_data.classes)):
            clientname_lists = [val['name'] for idx, val in enumerate(self.data_split['clients']) if label in val['data']]
            label_clientname_list.append(clientname_lists)
        self.data_split['label_clientname_list'] = label_clientname_list

    def get_clients_names(self):
        return self.clients_name

    def create_clients(self, unique_data: bool = True):
        """
        Splits the data for each client based on the JSON configuration.
        
        args: 
            - unique_data: If True, all clients get different data.
                            If False, clients can share the same data.
            - self.class_data: The training data with each class label.
            - self.data_split: The configuration for data splitting.
                - n_clients: The number of clients.
                - client_format: The format for client names.
                - class_clients_list: A list of clients for each class.
        
        return: a dictionary with keys clients' names and value as 
                data shards - tuple of images and label lists.
        """
        n_clients = self.data_split["n_clients"]
        client_format = self.data_split["client_format"]['name']

        # Clear previous client data
        self.clients = {f"{client_format}{i + 1}": [] for i in range(n_clients)}
        if unique_data:
            self._seperate_data_unique(self.train_data, self.clients)
        else:
            self._seperate_data_non_unique(self.train_data, self.clients)

        print(f"Data split successfully for {n_clients} clients.")
        return self.clients

    def _seperate_data_unique(self, train_data, clients):
        """
        Seperates the data for each client uniquely.
        
        args:
            - clients: A dictionary containing the clients and their data.
        
        return: A dictionary containing the clients and their data.
        """
        label_clientname_list = self.data_split['label_clientname_list']
        for idx, (image, label) in enumerate(train_data):
            client_names_list = label_clientname_list[label]
            if len(client_names_list) == 0:
                continue
            client_name = random.choice(client_names_list)
            self.clients[client_name].append(idx)
        return self.clients
    def _seperate_data_non_unique(self, train_data, clients):
        """
        Seperates the data for each client non-uniquely.
        
        args:
            - clients: A dictionary containing the clients and their data.
        
        return: A dictionary containing the clients and their data.
        """
        label_clientname_list = self.data_split['label_clientname_list']
        for idx, (image, label) in enumerate(train_data):
            client_names_list = label_clientname_list[label]
            for client_name in client_names_list:
                is_choose = random.choice([True, False])
                if is_choose:
                    self.clients[client_name].append(idx)
        return self.clients
            

    def _save_client_data_to_file(self):
        for client_name, data_indices in self.clients.items():
            print(f"Saving data for client: {client_name}")
            client_folder = os.path.join(self.data_dir, client_name)
            if not os.path.exists(client_folder):
                os.makedirs(client_folder)
            else: 
                # Clear previous data
                for file_name in os.listdir(client_folder):
                    os.remove(os.path.join(client_folder, file_name))
            
            for idx in data_indices:
                image, label = self.train_data[idx]
                image.save(os.path.join(client_folder, f"{idx}_{label}.PNG"))

        print("Client data saved successfully.")


    def generate_report(self) -> Dict:
        """
        Generates a report of the data distribution for each client.
        
        :return: A dictionary containing the number of samples for each client.
        """
        report = {client_name: {} for client_name in self.clients.keys()}
        for client_name, data_indices in self.clients.items():
            report[client_name]['num_data'] = len(data_indices)
            for label, class_name in enumerate(self.train_data.classes):
                report[client_name][class_name] = sum([1 for idx in data_indices if self.train_data.targets[idx] == label])

        with open(os.path.join(self.data_dir, 'data_split_report.json'), 'w') as f:
            json.dump(report, f, indent=4)
        return report

    def get_client_data(self, client_name: str):
        """
        Retrieves the dataset for a specific client.
        
        :param client_name: The name of the client.
        :return: A subset of the training data corresponding to the client's data.
        """
        if self.clients == None:
            return self.load_client_data(client_name)
        else: 
            client_data = [self.train_data[idx] for idx in self.clients[client_name]]
            return client_data
    def load_client_data(self, client_name: str):
        """
        Retrieves the dataset for a specific client from the saved files.
        
        :param client_name: The name of the client.
        :return: A subset of the training data corresponding to the client's data.
        """
        client_folder = os.path.join(self.data_dir, client_name)
        print(f"Loading data for client: {client_name} at {client_folder}")
        client_data = []
        for file_name in os.listdir(client_folder):
            # parse image to PIL.Image.Image format
            image = Image.open(os.path.join(client_folder, file_name))
            image = image.convert('RGB')
        
            label = int(file_name.split('_')[-1].split('.')[0])
            client_data.append((image, label))
        return client_data

    def get_clients_objects(self) -> List:
        clients = {}

        if(self.clients == None):
            for client in self.data_split['clients']:
                client_name = client['name']
                clients[client_name] = self.load_client_data(client_name)
        else:
            for client_name in self.clients.keys():
                clients[client_name] = self.get_client_data(client_name)
        return clients

def main():
    lab_folder = './'
    data_folder = os.path.join(lab_folder, 'data')

    def create_if_not_exist(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
    create_if_not_exist(data_folder)

    trainset = datasets.CIFAR10(root=data_folder, train=True, download=True)
    testset = datasets.CIFAR10(root=data_folder, train=False, download=True)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    nClasses  = 10

    dataSplit = DataSplitting(data_folder, os.path.join(data_folder, 'data_split.json'))
    dataSplit.create_clients(unique_data=True)
    dataSplit._save_client_data_to_file()
    dataSplit.generate_report()

# Run this code if this file is run as a script
if __name__ == "__main__":
    main()