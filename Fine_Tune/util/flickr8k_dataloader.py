from torch.utils.data import DataLoader, Dataset
import pickle
from PIL import Image
from torchvision import transforms
import torch
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
class Flickr8kDataset(Dataset):
    def __init__(self, folder_path: str, processed_file_path: str, transform: transforms.Compose, device: str):
        super().__init__()
        self.folder_path = folder_path
        self.transform = transform
        self.__processed_data: list[str, list[int]] = [] # list[image_id, tokenized_caption]
        self.device = device
        with open(processed_file_path, 'rb') as file:
            self.__processed_data = pickle.load(file)
        # print(len(set(data[0] for data in self.__processed_data)))
    def __len__(self):
        return len(self.__processed_data)
    def __getitem__(self, index):
        image_id, caption = self.__processed_data[index]
        image_path = os.path.join(self.folder_path, image_id+'.jpg')
        if not os.path.exists(image_path): raise Exception(f"File not found {image_path}")
        with Image.open(image_path).convert("RGB") as img:
            image = self.transform(img)
        caption = torch.tensor(caption.copy())
        return image.to(self.device), caption.to(self.device) , image_id

# if __name__ == "__main__":
def get_dataloader(
        folder_path: str, processed_file_path: str, transforms_: transforms.Compose, device: str, 
        batch_size: int, num_workers: int = 2, shuffle: bool = False):
    dataset = Flickr8kDataset(
        folder_path,
        processed_file_path,
        transforms_,
        device=device
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
