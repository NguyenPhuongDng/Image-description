from torch.utils.data import DataLoader, Dataset
import pickle
from PIL import Image
from torchvision import transforms
import torch
import os

class Flickr8kDataset(Dataset):
    def __init__(self, folder_path: str, processed_file_path: str, transform: transforms.Compose, device: str):
        super().__init__()
        self.folder_path = folder_path
        self.transform = transform
        self.__processed_data: list[str, str] = [] # list[image_id, tokenized_caption]
        self.device = device
        with open(processed_file_path, 'rb') as file:
            self.__processed_data = pickle.load(file)[:1000]
        print(len(set(data[0] for data in self.__processed_data)))
    def __len__(self):
        return len(self.__processed_data)
    def __getitem__(self, index):
        image_id, caption = self.__processed_data[index]
        image_path = os.path.join(self.folder_path, image_id+'.jpg')
        if not os.path.exists(image_path): raise Exception(f"File not found {image_path}")
        image = Image.open(image_path).convert("RGB")
        image: torch.Tensor = self.transform(image)
        caption = torch.tensor(caption)
        return image.to(self.device), caption.to(self.device)
    def get_dataloader(self, batch_size: int, num_workers: int = 2, shuffle: bool = False):
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)