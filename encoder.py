import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=True)
        
        # Remove last FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection layer
        self.projection = nn.Linear(2048, embed_dim)
        
    def forward(self, images):
        features = self.features(images)  # [batch, 2048, 7, 7]
        
        # Reshape to sequence of spatial features
        batch_size = features.size(0)
        features = features.view(batch_size, 2048, -1)  # [batch, 2048, 49]
        features = features.permute(0, 2, 1)  # [batch, 49, 2048]
        features = self.projection(features)  # [batch, 49, embed_dim]
        
        return features