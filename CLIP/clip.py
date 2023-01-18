import torch 
from torch import nn
import torch.nn.functional as F



class Attentionlayer(nn.Module):
    def __init__(self, embedding_size, num_heads) -> None:
        super().__init__()
        self.key = nn.Linear(embedding_size, embedding_size*num_heads)
        self.query = nn.Linear(embedding_size, embedding_size*num_heads)
        self.value = nn.Linear(embedding_size, embedding_size*num_heads)
    
        self.feed_forward = nn.Linear(embedding_size*num_heads, embedding_size)
        
        self.num_heads = num_heads
        self.embedding_size = embedding_size

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        z = F.softmax(q @ k.transpose(1,2), dim=-1) * v / (self.embedding_size ** .25) # as suggested in Transformers from scratch

        return self.feed_forward(z)


class AttentionBlock(nn.Module):
    def __init__(self, embedding_size, num_heads) -> None:
        super().__init__()
        self.attention_layer = Attentionlayer(embedding_size, num_heads)

        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            nn.ReLU(), #maybe GeLU
            nn.Linear(embedding_size * 4, embedding_size)
        )

    def forward(self, x):
        x_att = self.attention_layer(x)
        x_norm1 = self.norm1(x_att + x)
        x_ff = self.mlp(x_norm1)
        return self.norm2(x_ff + x_norm1)


class Transformer(nn.Module):
    def __init__(self, embedding_size, seq_length, num_heads, n_blocks) -> None:
        super().__init__()
        self.positional_encoding = nn.Parameters(torch.randn(seq_length, embedding_size))
        # create list of attention block
    def forward(self, x):
        #loop over the attention blocks
        x = x + self.positional_encoding
        pass

class VisionTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #conv layer to embed patches
        #classical transformer afterward
        #may need 2 layer normalizations (pre and post transformer as in openai)

    def forward(self):
        pass


class Clip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #initialize text and image encoders
        #similarity matrix

    def forward_text(self):
        pass

    def forward_image(self):
        pass

    def forward(self):
        pass