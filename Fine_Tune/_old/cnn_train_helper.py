import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
import os
import tqdm
import matplotlib.pyplot as plt
import datetime
from typing import Type
import time
def save_model(model: nn.Module, folder_path: str, time_stamp: str, name: str, logs: dict | list, metadata: dict[str, object]):
    folder_path = os.path.join(folder_path, time_stamp)
    file_path = os.path.join(folder_path, f"{name}.pt")
    if not os.path.exists(folder_path): os.makedirs(folder_path)
    torch.save(model, file_path)
    file_path = os.path.join(folder_path, "logs.json")
    import json
    with open(file_path, 'w') as file:
        file.write(json.dumps(logs))
    file_path = os.path.join(folder_path, "metadata.json")
    with open(file_path, 'w') as file:
        file.write(json.dumps(metadata))
    file_path = os.path.join(folder_path, "model_info.txt")
    with open(file_path, 'w') as file:
        file.write(str(model))


def train(model : nn.Module, dataloader : DataLoader, lossf : callable, optimizer : torch.optim.Optimizer, mixed: bool, device: str):
    model.train()
    epoch_loss = 0
    count = 0
    for data in dataloader:
        optimizer.zero_grad()
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        if mixed:
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = model(inputs)
                loss: torch.Tensor= lossf(outputs, labels)
        else:
            outputs = model(inputs)
            loss: torch.Tensor= lossf(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        count += 1
    return epoch_loss/count
def test(model : nn.Module, dataloader : DataLoader, lossf : callable, mixed: bool, device: str):
    model.eval()
    epoch_loss = 0
    correct_preds = 0
    count = 0
    total_preds = 0
    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        if mixed:
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = model(inputs)
                loss: torch.Tensor= lossf(outputs, labels)
        else:
            outputs = model(inputs)
            loss: torch.Tensor= lossf(outputs, labels)
        epoch_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct_preds += (preds == labels).sum().item()
        count += 1
        total_preds += labels.shape[0]
    return epoch_loss/count, correct_preds / total_preds

def train_eval(
        trainloader : DataLoader, 
        testloader : DataLoader, 
        model : nn.Module, 
        lossf : callable,
        num_epochs : int,
        lr : float, 
        gamma : float = 1, 
        log_step : int = 5, 
        warmup_nepochs : int = 0, 
        warmup_lr : float = 0.1,
        warmup_gamma : float = 1,
        save: bool = False,
        save_path: str = "train_log",
        mixed: bool = False,
        optimizer_type: Type[torch.optim.Adam] | Type[torch.optim.SGD]  = torch.optim.Adam,
        metadata_extra: dict[str, str] = {},
        device: str = 'cuda'
    ):
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    start_time = time.time()
    train_loader = trainloader
    test_loader = testloader
    optimizer = optimizer_type(model.parameters(),lr=warmup_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    warmup_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, warmup_gamma)
    model.to(device)
    print("Start train")
    train_losses = []
    test_losses = []
    test_accuracies = []
    lrs = []
    def get_train_log():
        training_logs = [{
            "Epoch" : i + 1,
            "Train loss" : train_losses[i],
            "Test loss" : test_losses[i],
            "Test accuracy" : test_accuracies[i],
            "Learning rate" : lrs[i]
        } for i in range(len(lrs))]
        return training_logs
    def get_hyper_parameter():
        hyper_parameters = {
            'nepochs' : num_epochs,
            'lr' : lr,
            'gamma' : gamma,
            'optimizer' : optimizer_type.__name__,
            'warmup_nepochs' : warmup_nepochs,
            'warmup_lr' : warmup_lr,
            'warmup_gamma' : warmup_gamma,
            'mixed' : mixed,
            'lossf' : lossf._get_name(),
            'total_train_eval_time' : time.time() - start_time
        }
        hyper_parameters.update(metadata_extra)
        return hyper_parameters
    for epoch in tqdm.trange(num_epochs):
        if epoch >= warmup_nepochs: lrs.append(scheduler.get_last_lr()[0])
        else: lrs.append(warmup_scheduler.get_last_lr()[0])
        if (epoch == warmup_nepochs):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if len(lrs) > 0: lrs[-1] = lr
        train_losses.append(train(model, train_loader, lossf, optimizer, mixed, device))
        test_loss, test_accuracy = test(model, test_loader, lossf, mixed, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        if (epoch >= warmup_nepochs): scheduler.step()
        else: warmup_scheduler.step()
        if ((epoch + 1) % log_step == 0):
            print(f"Train loss : {train_losses[-1]:.4f} | Test loss : {test_losses[-1]:.4f} | Test accuracy : {test_accuracies[-1]:.4f} | Lr : {lrs[-1]}")
    if save:
        save_model(model, save_path, time_stamp, f"model", get_train_log(), get_hyper_parameter())
    plt.figure(figsize=(10,6))
    plt.plot(train_losses, label = "Train loss", color = "blue")
    plt.plot(test_losses, label = "Test loss", color = "red")
    plt.legend()
    plt.figure(figsize=(10,6))
    plt.plot(test_accuracies, label= "Test accuracy", color = "blue")
    plt.legend()
    plt.figure(figsize=(10,6))
    plt.plot(lrs, label= "Learning rate", color = "blue")
    plt.legend()
    print("Complete")
    return os.path.join(save_path, time_stamp, f"model.pt")