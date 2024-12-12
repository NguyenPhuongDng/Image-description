import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
import os
import tqdm
import matplotlib.pyplot as plt
import datetime
from typing import Type, Callable
import time
from .model_serilizer import save_model_chunks, load_model_chunks, save_state_dicts, load_state_dicts

def save_model(model: nn.Module, folder_path: str, name: str, logs: dict | list, metadata: dict[str, object], max_chunk_size : int):
    file_path = os.path.join(folder_path, f"{name}.pt")
    if not os.path.exists(folder_path): os.makedirs(folder_path)
    torch.save(model, file_path)
    save_model_chunks(model, folder_path, name, max_chunk_size)
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

def train_eval(
        trainloader : DataLoader, 
        testloader : DataLoader, 
        model : nn.Module,
        train_func: Callable[[nn.Module, DataLoader, Callable[[any], torch.Tensor], torch.optim.Optimizer, bool, str], float],
        test_func: Callable[[nn.Module, DataLoader, Callable[[any], torch.Tensor], bool, str], dict[str, object]],
        lossf : callable,
        num_epochs : int,
        lr : float, 
        gamma : float = 1, 
        log_step : int = 5, 
        warmup_nepochs : int = 0, 
        warmup_lr : float = 0.1,
        warmup_gamma : float = 1,
        load_checkpoint: bool = False,
        load_optimizer: bool = False,
        checkpoint_path: str = None,
        save: bool = False,
        save_optimizer: bool = False,
        save_each: int = -1,
        save_path: str = "train_log",
        mixed_train: bool = False,
        mixed_eval: bool = False,
        optimizer_type: Type[torch.optim.Adam] | Type[torch.optim.SGD]  = torch.optim.Adam,
        metadata_extra: dict[str, str] = {},
        device: str = 'cuda',
        log_metric: bool = False,
        max_model_size: int = 99 * 1024 * 1024
    ):
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    start_time = time.time()
    train_loader = trainloader
    test_loader = testloader
    if load_checkpoint:
        print("Load checkpoint")
        model = load_model_chunks(checkpoint_path, "model")
        optimizer = optimizer_type(model.parameters(),lr=warmup_lr)
        if load_optimizer:
            print("Load optimizer")
            optimizer.load_state_dict(load_state_dicts(checkpoint_path, "optimizer")['data'])
    else:
        optimizer = optimizer_type(model.parameters(),lr=warmup_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    warmup_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, warmup_gamma)
    model.to(device)
    print("Start train")
    train_losses = []
    test_losses = []
    test_metrics = []
    lrs = []
    total_train_time = 0
    def save_checkpoint(sub_folder: str = None):
        if sub_folder:
            real_folder = os.path.join(save_path, time_stamp, sub_folder)
        else:
            real_folder = os.path.join(save_path, time_stamp)
        save_model(model, real_folder, f"model", get_train_log(), get_hyper_parameter(), max_model_size)
        if save_optimizer:
            save_state_dicts(optimizer.state_dict(), real_folder, f"optimizer", float('inf'))
    def get_train_log():
        training_logs = [{
            "Epoch" : i + 1,
            "Train loss" : train_losses[i],
            "Test loss" : test_losses[i],
            "Learning rate" : lrs[i],
            "Metrics" : test_metrics[i]
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
            'mixed_train' : mixed_train,
            'mixed_eval' : mixed_eval,
            'lossf' : lossf._get_name(),
            'total_train_eval_time' : time.time() - start_time,
            'total_train_time' : total_train_time,
            'total_eval_and_control_time' : time.time() - start_time - total_train_time
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
        train_start_time = time.time()
        train_losses.append(train_func(model, train_loader, lossf, optimizer, mixed_train, device))
        epoch_train_time = time.time() - train_start_time
        total_train_time += epoch_train_time
        epoch_metrics = test_func(model, test_loader, lossf, mixed_eval, device)
        test_loss = epoch_metrics.get("loss", 0)
        test_metrics.append(epoch_metrics)
        test_losses.append(test_loss)
        if (epoch >= warmup_nepochs): scheduler.step()
        else: warmup_scheduler.step()
        if ((epoch + 1) % log_step == 0):
            print(f"Train loss : {train_losses[-1]:.4f} | Test loss : {test_losses[-1]:.4f} | Train time : {epoch_train_time:.2f} s | Lr : {lrs[-1]:.8f}")
            if log_metric: print(epoch_metrics)
        if (save_each != -1 and (epoch +1) % save_each  == 0 and (epoch + 1) != num_epochs):
            save_checkpoint(f"E{epoch+1}")
    if save:
        save_checkpoint()
    print("Complete")
    return os.path.join(save_path, time_stamp, f"model.pt"), get_train_log()