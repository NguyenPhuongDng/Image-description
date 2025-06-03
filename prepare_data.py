import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import json
import os
from PIL import Image
from torchvision import transforms

class FlickrDataset(Dataset):
    """Dataset class cho Flickr8k Image Captioning"""
    
    def __init__(self, data_dir='Processed Data', split='train', image_dir='Flickr8k/Flicker8k_Dataset', 
                 transform=None):
        """
        Args:
            data_dir: Thư mục chứa processed data
            split: 'train', 'val', hoặc 'test'
            image_dir: Thư mục chứa ảnh gốc
            transform: Transform cho ảnh
            load_images: True nếu muốn load ảnh, False nếu chỉ cần image paths
        """
        self.data_dir = data_dir
        self.split = split
        self.image_dir = image_dir
        self.transform = transform
        
        # Load vocab
        self.load_vocab()
        
        # Load metadata
        self.load_metadata()
        
        # Load sequences
        self.load_sequences()
        
        # Tạo danh sách (image_id, caption_idx) pairs
        self.create_data_pairs()
        
        print(f"FlickrDataset {split} loaded:")
        print(f"  Images: {len(self.sequences)}")
        print(f"  Image-caption pairs: {len(self.data_pairs)}")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Max length: {self.max_length}")
    
    def load_vocab(self):
        """Load vocabulary mappings"""
        vocab_path = os.path.join(self.data_dir, 'vocab.json')
        decode_vocab_path = os.path.join(self.data_dir, 'decode_vocab.json')
        
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        
        with open(decode_vocab_path, 'r') as f:
            self.decode_vocab = json.load(f)
            # Convert keys to int
            self.decode_vocab = {int(k): v for k, v in self.decode_vocab.items()}
        
        # Special tokens
        self.pad_token = self.vocab['<PAD>']
        self.start_token = self.vocab['<START>']
        self.end_token = self.vocab['<END>']
        self.unk_token = self.vocab['<UNK>']
    
    def load_metadata(self):
        """Load metadata"""
        metadata_path = os.path.join(self.data_dir, 'metadata.json')
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.vocab_size = metadata['vocab_size']
        self.max_length = metadata['max_length']
        self.min_freq = metadata['min_freq']
    
    def load_sequences(self):
        """Load caption sequences cho split"""
        sequences_path = os.path.join(self.data_dir, f'{self.split}_sequences.pkl')
        
        with open(sequences_path, 'rb') as f:
            self.sequences = pickle.load(f)
    
    def create_data_pairs(self):
        """Tạo danh sách (image_id, caption_idx) pairs"""
        self.data_pairs = []
        
        for image_id, captions in self.sequences.items():
            for caption_idx in range(len(captions)):
                self.data_pairs.append((image_id, caption_idx))
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        image_id, caption_idx = self.data_pairs[idx]
        
        # Get caption sequence
        caption = self.sequences[image_id][caption_idx]
        caption_tensor = torch.tensor(caption, dtype=torch.long)
        
        # Get image
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        return {
            'image': image,
            'caption': caption_tensor,
            'image_id': image_id,
            'caption_idx': caption_idx
        }
    
    def decode_caption(self, sequence):
        """Decode sequence thành text"""
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.tolist()
        
        words = []
        for token_id in sequence:
            word = self.decode_vocab.get(token_id, '<UNK>')
            if word not in ['<PAD>', '<START>', '<END>']:
                words.append(word)
            elif word == '<END>':
                break
        
        return ' '.join(words)
    
    def get_vocab_info(self):
        """Return vocab info for model"""
        return {
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'vocab': self.vocab,
            'decode_vocab': self.decode_vocab,
            'pad_token': self.pad_token,
            'start_token': self.start_token,
            'end_token': self.end_token,
            'unk_token': self.unk_token
        }

def get_image_transforms():
    """Standard image transforms cho ResNet"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def create_flickr_dataloaders(data_dir='Processed Data', image_dir='Flickr8k/Flicker8k_Dataset',
                             batch_size=32, num_workers=0):
    """Tạo DataLoaders cho train/val/test"""
    
    # Image transforms
    transform = get_image_transforms()
    
    # Create datasets
    train_dataset = FlickrDataset(
        data_dir=data_dir, 
        split='train', 
        image_dir=image_dir,
        transform=transform
    )
    
    val_dataset = FlickrDataset(
        data_dir=data_dir, 
        split='val', 
        image_dir=image_dir,
        transform=transform
    )
    
    test_dataset = FlickrDataset(
        data_dir=data_dir, 
        split='test', 
        image_dir=image_dir,
        transform=transform
    )

    # Custom collate function
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        captions = torch.stack([item['caption'] for item in batch])
        image_ids = [item['image_id'] for item in batch]
        caption_indices = [item['caption_idx'] for item in batch]
        
        return {
            'images': images,
            'captions': captions,
            'image_ids': image_ids,
            'caption_indices': caption_indices
        }
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader, train_dataset.get_vocab_info()

# Example usage
def main():
    """Test FlickrDataset"""
    print("Testing FlickrDataset...")
    
    # Test without loading images (faster)
    train_loader, val_loader, test_loader, vocab_info = create_flickr_dataloaders(
        batch_size=32
    )
    
    print(f"\nVocab info:")
    print(f"  Vocab size: {vocab_info['vocab_size']}")
    print(f"  Max length: {vocab_info['max_length']}")
    
    # Test một batch
    print(f"\nTesting data loading...")
    for batch in train_loader:
        print(f"Batch info:")
        print(f"  Images: {len(batch['images'])} paths")
        print(f"  Captions shape: {batch['captions'].shape}")
        print(f"  Sample image ID: {batch['image_ids'][0]}")
        
        # Decode một caption
        dataset = train_loader.dataset
        sample_caption = dataset.decode_caption(batch['captions'][0])
        print(f"  Sample caption: {sample_caption}")
        
        break
    
    print("\nFlickrDataset test completed!")

if __name__ == "__main__":
    main()