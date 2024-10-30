import os
from time import time
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

from DataSplitting import DataSplitting
from Model import CIFAR10

print('Start Program')
lab_folder = './'
model_folder = os.path.join(lab_folder, 'model')
data_folder = os.path.join(lab_folder, 'data')

def create_if_not_exist(folder):
  if not os.path.exists(folder):
    os.makedirs(folder)
create_if_not_exist(model_folder)
create_if_not_exist(data_folder)


trainset = torchvision.datasets.CIFAR10(root=data_folder, train=True, download=True)
testset = torchvision.datasets.CIFAR10(root=data_folder, train=False, download=True)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
nClasses  = 10

def get_data_distribution(dataSplit):
    dataSplit.create_clients(unique_data=True)
    dataSplit._save_client_data_to_file()
    dataSplit.generate_report()
    return dataSplit

def transform_data(data):
  transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  data = [(transform(image), label) for image, label in data]
  return data

batch_size = 32
pCV = 0.1
def splitIndices(m, pCV):
  """ randomly shuffle a training set's indices, then split the indices into training and cross validation sets.
   Pass in 'm', length of training set, and 'pCV', the percentage of the training set you would like 
   to dedicate to cross validation."""
   
  mCV = int(m*pCV)
  indices = np.random.permutation(m)
  return indices[mCV:], indices[:mCV]

def prepare_data(client_data):
    indices = list(range(len(client_data)))
    trainIndices, valIndices = splitIndices(len(client_data), pCV)

    trainSampler = SubsetRandomSampler(indices)
    trainLoader = DataLoader(client_data, batch_size=batch_size, sampler=trainSampler, drop_last=True)

    validSampler = SubsetRandomSampler(valIndices)
    validLoader = DataLoader(client_data, batch_size=batch_size, sampler=validSampler, drop_last=True)
    return trainLoader, validLoader

lossFn = F.cross_entropy

def lossBatch(model, lossFn, xb, yb, opt=None, metric=None):
  preds = model(xb)
  loss = lossFn(preds, yb)

  if opt is not None:
    loss.backward()
    # update parameters
    opt.step()
    # reset gradients to 0 (don't want to calculate second derivatives!)
    opt.zero_grad()

  metricResult = None
  if metric is not None:
    metricResult = metric(preds, yb)

  return loss.item(), len(xb),  metricResult

def evaluate(model, lossFn, validDL, metric=None):
  results = [lossBatch(model, lossFn, xb, yb, metric=metric,) for xb,yb in validDL]

  losses, nums, metrics = zip(*results)

  total = np.sum(nums)  # size of the dataset

  avgLoss = np.sum(np.multiply(losses, nums))/total

  # if there is a metric passed, compute the average metric
  if metric is not None:
    avgMetric = np.sum(np.multiply(metrics, nums)) / total

  return avgLoss, total, avgMetric

def accuracy(outputs, labels):
  _, preds = torch.max(outputs, dim=1) # underscore discards the max value itself, we don't care about that
  return torch.sum(preds == labels).item() / len(preds)

def fit(epochs, model, lossFn, opt, trainDL, valDL, metric=None, patience=5):
  best_val_loss = float('inf')
  best_weights = model.state_dict()
  patience_counter = 0

  valList = [0.10]
  for epoch in range(epochs):
    # training - perform one step gradient descent on each batch, then moves on
    for xb, yb in trainDL:
      loss, _, lossMetric = lossBatch(model, lossFn, xb, yb, opt)

    with torch.no_grad():  # Disable gradient calculation for validation
        valResult = evaluate(model, lossFn, valDL, metric)
        valLoss, total, valMetric = valResult
        valList.append(valMetric)

    # Check if the current epoch's validation loss is the best
    if valLoss < best_val_loss:
        best_val_loss = valLoss
        best_weights = model.state_dict()  # Save the best model weights
        patience_counter = 0  # Reset the patience counter
    else:
        patience_counter += 1  # Increment patience counter if no improvement

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        model.load_state_dict(best_weights)  # Load the best weights
        break

    # print progress
    if metric is None:
      print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, valLoss))
    else:
      print('Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}'.format(epoch + 1, epochs, valLoss, metric.__name__, valMetric))

  model.load_state_dict(best_weights)
  return valList


def evaluate_with_dataset(model, datasetT, name="Data"):
    X, y = zip(*datasetT)
    X = np.array(X)
    y = np.array(y)

    y_pred = model.predict(torch.tensor(X, dtype=torch.float32))
    print(f"{name} accuracy: %.3f" % accuracy_score(y, y_pred))
    ConfusionMatrixDisplay.from_predictions(y, y_pred)
    plt.show()

epochs = 10
def train_local_model(client_data, global_config):
    trainLoader, validLoader = prepare_data(client_data)
    model = CIFAR10(global_config['inputShape'], global_config['outputShape'])
    if global_config['state_dict'] :
        model.load_state_dict(global_config['state_dict'])
        model.to(torch.float32)

    dataLen = len(client_data)

    learningRate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    metric = accuracy
    fit(epochs, model, lossFn, optimizer, trainLoader, validLoader, metric)

    return model.state_dict(), dataLen

def weight_scaling_factor(client_results):
    # Calculate the total data points across all clients
    totalDataPoints = sum([client_results[client][1] for client in client_results])
    # Calculate the scaling factor for each client
    scaling_factor = {client: (client_results[client][1] / totalDataPoints) for client in client_results}
    return scaling_factor

def scale_model_state_dict(state_dict, scale_factor):
    scaled_state_dict = {}
    for key in state_dict:
        scaled_state_dict[key] = state_dict[key] * scale_factor
    return scaled_state_dict

def aggregate_state_dicts(scaled_state_dicts):
    aggregated_state_dict = {}
    for key in scaled_state_dicts[0]:  # Initialize keys from first client
        stacked_tensors = torch.stack([client_dict[key] for client_dict in scaled_state_dicts])
        aggregated_state_dict[key] = stacked_tensors.sum(dim=0)
    return aggregated_state_dict

def federated_averaging(client_results):
    # Calculate the scaling factor for each client
    scaling_factor = weight_scaling_factor(client_results)

    # Scale each client's state_dict by its scaling factor
    scaled_state_dicts = []
    for client in client_results:
        state_dict = client_results[client][0]
        scaled_state_dict = scale_model_state_dict(state_dict, scaling_factor[client])
        scaled_state_dicts.append(scaled_state_dict)

    # Average the scaled state_dicts
    aggregated_state_dict = aggregate_state_dicts(scaled_state_dicts)
    return aggregated_state_dict

def federated_averaging_v2(client_results):
    # Initialize an empty state dict to hold the averaged weights
    aggregated_state_dict = {}

    # Get the state dicts from the first client to initialize the keys
    first_client_key = next(iter(client_results))
    aggregated_state_dict = {key: torch.zeros_like(client_results[first_client_key][key]) for key in client_results[first_client_key]}

    # Sum the weights from all clients
    for client_name, state_dict in client_results.items():
        for key in state_dict:
            aggregated_state_dict[key] += state_dict[key]

    # Average the weights by dividing by the number of clients
    num_clients = len(client_results)
    for key in aggregated_state_dict:
        aggregated_state_dict[key] /= num_clients

    return aggregated_state_dict

def main():
    dataSplitting = DataSplitting(data_folder, os.path.join(data_folder, 'data_split.json'))
    dataSplitting.load_report(os.path.join(data_folder, 'data_split_report.json'))
    # dataSplitting = get_data_distribution(dataSplitting)
    clients_name = dataSplitting.get_clients_names()

    # load and transform data
    clients = {}
    for client_name in clients_name:
        clients[client_name] = transform_data(dataSplitting.get_client_data(client_name))
    # server do it 
    # server do it
    testData = transform_data(testset)
    testDL = DataLoader(testData, batch_size=batch_size)

    global_config = {
        'inputShape': 3 * 32 * 32,
        'outputShape': 10,
        'state_dict': None
    }
    comms_round = 20
    global_model = CIFAR10(global_config['inputShape'], global_config['outputShape'])

    best_loss = float('inf')
    best_model_state = global_model.state_dict()
    patience = 0  # Number of rounds to wait for an improvement in accuracy
    max_patience = 2  # Set the maximum number of rounds for early stopping

    # Lists to save data for reporting
    losses = []
    accuracies = []

    for comm_round in range(comms_round):
        print(f"---Starting round {comm_round}---")
        global_state_dict = global_model.state_dict()
        global_config['state_dict'] = global_state_dict
        client_results = {}

        for client_name in clients_name:
            print(f"\tTraining {client_name}")
            client_data = clients[client_name]
            state_dict, dataLen = train_local_model(client_data, global_config)
            client_results[client_name] = (state_dict, dataLen)
            print(f"Client {client_name} has trained their model with {dataLen} data points.")

        aggregated_state_dict = federated_averaging(client_results)
        global_model.load_state_dict(aggregated_state_dict)
        print(f"Round {comm_round} has completed.")

        # Evaluate the global model
        loss, _, acc = evaluate(global_model, lossFn, testDL, metric=accuracy)
        losses.append(loss)
        accuracies.append(acc)
        print(f"Test loss: {loss}, Test accuracy: {acc}")

        # Save the best model based on the lowest loss
        if loss < best_loss:
            best_loss = loss
            best_model_state = global_model.state_dict()
            patience = 0  # Reset patience if we find a new best model
            print(f"New best model found with loss: {best_loss}")
        else:
            patience += 1  # Increment patience if no improvement

        # Early stopping if accuracy does not increase
        if patience >= max_patience:
            print("Early stopping: no improvement in accuracy.")
            break

        # Save the best model state after training
        global_model.load_state_dict(best_model_state)
        global_model.save(os.path.join(model_folder, f'global_model_lost_{best_loss}.pth'))
        print(f"Model saved with loss: {best_loss}")
        print("Training completed.")
    
main()