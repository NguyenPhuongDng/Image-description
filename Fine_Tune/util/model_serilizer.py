import torch
import torch.nn as nn
import os
import io

def save_model_chunks(model: nn.Module, folder_path: str, name: str, max_chunk_size: int):
    file_path = os.path.join(folder_path, f"{name}.pt")
    if not os.path.exists(folder_path): os.makedirs(folder_path)
    torch.save(model, file_path)
    model_size = os.path.getsize(file_path)
    if model_size > max_chunk_size:
        with open(file_path, 'rb') as file:
            model_bytes = file.read()
        num_chunks = (len(model_bytes) // max_chunk_size) + 1
        chunk_size = len(model_bytes) // num_chunks
        for i in range(num_chunks):
            start_index = i * chunk_size
            end_index = min((i + 1) * chunk_size, len(model_bytes))
            chunk_file_name = os.path.join(folder_path, f"{name}{i}.pt")
            with open(chunk_file_name, 'wb') as file:
                file.write(model_bytes[start_index:end_index])
        os.remove(file_path)

def load_model_chunks(folder_path: str, name: str):
    files = os.listdir(folder_path)
    i = 0
    if f'{name}.pt' in files:
        return torch.load(os.path.join(folder_path, f'{name}.pt'), weights_only=False)
    file_paths = []
    while True:
        if f'{name}{i}.pt' in files:
            file_paths.append(os.path.join(folder_path, f'{name}{i}.pt'))
            i+=1
        else:
            break
    model_bytes = b''
    for file_path in file_paths:
        with open(file_path, 'rb') as file:
            model_bytes += file.read()
    temp_file = os.path.join(folder_path, "temp.pt")
    with open(temp_file, 'wb') as file:
        file.write(model_bytes)
    model = torch.load(temp_file, weights_only=False)
    os.remove(temp_file)
    return model

def save_state_dicts(state_dict, folder_path: str, name: str, max_chunk_size: int):
    file_path = os.path.join(folder_path, f"{name}.pt")
    if not os.path.exists(folder_path): os.makedirs(folder_path)
    torch.save({"data" : state_dict}, file_path)
    model_size = os.path.getsize(file_path)
    # if model_size > max_chunk_size:
    #     with open(file_path, 'rb') as file:
    #         model_bytes = file.read()
    #     num_chunks = (len(model_bytes) // max_chunk_size) + 1
    #     chunk_size = len(model_bytes) // num_chunks
    #     for i in range(num_chunks):
    #         start_index = i * chunk_size
    #         end_index = min((i + 1) * chunk_size, len(model_bytes))
    #         chunk_file_name = os.path.join(folder_path, f"{name}{i}.pt")
    #         with open(chunk_file_name, 'wb') as file:
    #             file.write(model_bytes[start_index:end_index])
    #     os.remove(file_path)
def load_state_dicts(folder_path: str, name: str):
    files = os.listdir(folder_path)
    i = 0
    if f'{name}.pt' in files:
        return torch.load(os.path.join(folder_path, f'{name}.pt'), weights_only=False)
    file_paths = []
    while True:
        if f'{name}{i}.pt' in files:
            file_paths.append(os.path.join(folder_path, f'{name}{i}.pt'))
            i+=1
        else:
            break
    model_bytes = b''
    for file_path in file_paths:
        with open(file_path, 'rb') as file:
            model_bytes += file.read()
    data = torch.load(io.BytesIO(model_bytes), weights_only=True)["data"]
    return data



