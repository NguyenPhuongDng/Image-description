import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].permute(1, 0, 2)

class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, captions, image_features):
        # Token embeddings
        caption_embeds = self.token_embedding(captions)
        caption_embeds = self.pos_encoding(caption_embeds)
        
        # Image features đã là [batch, 49, embed_dim] từ encoder
        # Không cần unsqueeze(1) nữa
        memory = image_features  # [batch, 49, embed_dim]
        
        # Causal mask để không nhìn tương lai
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(captions.size(1)).to(captions.device)
        
        # Decoder forward
        output = self.transformer_decoder(caption_embeds, memory, tgt_mask=tgt_mask)
        logits = self.output_projection(output)
        
        return logits