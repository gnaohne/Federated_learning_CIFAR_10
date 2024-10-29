import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.preprocessing import LabelBinarizer
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

def get_data_distribution(dataset):
    dataSplit = DataSplitting(data_folder, os.path.join(data_folder, 'data_split.json'))
    dataSplit.create_clients(unique_data=True)
    dataSplit._save_client_data_to_file()
    dataSplit.generate_report()
    return dataSplit

def transform_to_tensor(imageRGB):
    return  torch.tensor(np.array(imageRGB))

def transform_data(data):
    images, labels = zip(*data)
    images, labels = list(images), list(labels)
    images = [transform_to_tensor(image) for image in images]
    images = [image.float() for image in images]  # convert to float

    labels = torch.tensor(labels)  # Convert labels directly to tensor without one-hot encoding

    return list(zip(images, labels))

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
#   best_val_loss = float('inf')
#   best_weights = model.state_dict()
#   patience_counter = 0

  valList = [0.10]
  for epoch in range(epochs):
    # training - perform one step gradient descent on each batch, then moves on
    for xb, yb in trainDL: 
      loss, _, lossMetric = lossBatch(model, lossFn, xb, yb, opt)
      

    # evaluation on cross val dataset - after updating over all batches, technically one epoch
    # evaluates over all validation batches and then calculates average val loss, as well as the metric (accuracy)
    valResult = evaluate(model, lossFn, valDL, metric)
    valLoss, total, valMetric = valResult
    valList.append(valMetric)

    # model.eval()  # Ensure model is in evaluation mode
    # with torch.no_grad():  # Disable gradient calculation for validation
    #     valResult = evaluate(model, lossFn, valDL, metric)
    #     valLoss, total, valMetric = valResult
    #     valList.append(valMetric)

    # # Check if the current epoch's validation loss is the best
    # if valLoss < best_val_loss:
    #     best_val_loss = valLoss
    #     best_weights = model.state_dict()  # Save the best model weights
    #     patience_counter = 0  # Reset the patience counter
    # else:
    #     patience_counter += 1  # Increment patience counter if no improvement

    # print progress
    if metric is None: 
      print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, valLoss))
    else:
      print('Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}'.format(epoch + 1, epochs, valLoss, metric.__name__, valMetric))

  return valList

epochs = 10
def train_local_model(client_data, initWeights=None):
    trainLoader, validLoader = prepare_data(client_data)
    model = CIFAR10()
    dataLen = len(client_data)

    if initWeights is not None:
        model.set_weights(initWeights)
    
    learningRate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    metric = accuracy
    fit(epochs, model, lossFn, optimizer, trainLoader, validLoader, metric)
    return model.get_weights(), dataLen

def weight_scalling_factor(client_results):
    # calculate the total data points across all clients
    totalDataPoints = sum([client_results[client][1] for client in client_results])
    # calculate the scaling factor
    scaling_factor = {client: (client_results[client][1] / totalDataPoints) for client in client_results}
    return scaling_factor

def sum_scaled_weights(scaled_weights):
    # each weight is tensor [10, 3072]
    avg_weight = list()
    for weights_list_tuple in zip(*scaled_weights):
        layer_mean = torch.stack(weights_list_tuple).mean(0)
        avg_weight.append(layer_mean)
    avg_weight = torch.stack(avg_weight)
    return avg_weight
    
def scale_model_weights(client_results, scaling_factor):
    scaled_weights = []
    for client in scaling_factor:
        weight = client_results[client][0]
        scaled_weight = [scaling_factor[client] * w for w in weight]
        scaled_weight = torch.stack(scaled_weight)
        scaled_weights.append(scaled_weight)
    return scaled_weights

def aggregate_model_weights(client_results):
    scaling_factor = weight_scalling_factor(client_results)
    scaled_weights = scale_model_weights(client_results, scaling_factor)
    average_weight = sum_scaled_weights(scaled_weights)
    return average_weight

def aggregate_model_weights_v2(client_results):
    pass

def main():
    dataSplitting = DataSplitting(data_folder, os.path.join(data_folder, 'data_split.json'))
    dataSplitting.create_clients(unique_data=True)
    dataSplitting._save_client_data_to_file()
    dataSplitting.generate_report()

    
    clients_name = dataSplitting.get_clients_names()

    # load and transform data
    clients = {}
    for client_name in clients_name:
        clients[client_name] = transform_data(dataSplitting.get_client_data(client_name))
    # server do it 
    testData = transform_data(testset)
    testDL = DataLoader(testData, batch_size=batch_size)
        
    comms_round = 5
    global_model = CIFAR10()
    global_model.set_weights(global_model.get_weights())

    for comm_round in range(comms_round):
        print(f"---Starting round {comm_round}---")
        global_weights = global_model.get_weights()
        client_results = {}
        
        for client_name in clients_name:
            print(f"\tTraining {client_name}")
            client_data = clients[client_name]
            weights, dataLen = train_local_model(client_data)
            client_results[client_name] = (weights, dataLen)
            print(f"Client {client_name} has trained their model with {dataLen} data points.")
        
        # average the weights
        aggregated_weights = aggregate_model_weights(client_results)
        global_model.set_weights(aggregated_weights)
        print(f"Round {comm_round} has completed.")
        evaluate(global_model, lossFn, testDL, metric=accuracy)
    
main()