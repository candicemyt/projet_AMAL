import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np


class Attentionlayer(nn.Module):
    def __init__(self, embedding_size, num_heads) -> None:
        super().__init__()
        self.key = nn.Linear(embedding_size, embedding_size*num_heads)
        self.query = nn.Linear(embedding_size, embedding_size*num_heads)
        self.value = nn.Linear(embedding_size, embedding_size*num_heads)
    
        self.feed_forward = nn.Linear(embedding_size*num_heads, embedding_size)
        
        self.num_heads = num_heads
        self.embedding_size = embedding_size

    def forward(self, x, mask):
        q = self.query(x) #b grid**2 embedding_size*num_heads
        k = self.key(x)
        v = self.value(x)
        i = q @ k.transpose(1,2)

        if mask is not None:
            z = torch.bmm(F.softmax(i+mask, dim=-1),v) / (self.embedding_size ** .25) # as suggested in Transformers from scratch
        else: 
            z = torch.bmm(F.softmax(i, dim=-1), v) / (self.embedding_size ** .25) # as suggested in Transformers from scratch

        return self.feed_forward(z)


class AttentionBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, mask=None) -> None:
        super().__init__()
        self.attention_layer = Attentionlayer(embedding_size, num_heads)

        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            nn.ReLU(), #maybe GeLU
            nn.Linear(embedding_size * 4, embedding_size)
        )

        self.mask = mask

    def attention(self, x):
        return self.attention_layer(x, self.mask)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, embedding_size, seq_length, num_heads, n_blocks, vocab_size, output_dim, use_mask=True) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_blocks = n_blocks
        self.num_heads = num_heads
        self.seq_length = seq_length

        self.positional_encoding = nn.Parameter(torch.empty(seq_length, embedding_size))
        self.token_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        self.mask = self.build_mask() if use_mask else None

        # create list of attention block
        attention_blocks = []
        for i in range(n_blocks):
            attention_blocks.append(AttentionBlock(embedding_size=embedding_size, num_heads=num_heads, mask=self.mask))
        self.attention_blocks = nn.Sequential(*attention_blocks)

        self.linear = nn.Linear(seq_length, 1)
        self.norm = nn.LayerNorm(embedding_size)
        self.proj = nn.Parameter(torch.rand(embedding_size, output_dim))

    def build_mask(self):
        mask = torch.empty(self.seq_length, self.seq_length)
        mask.fill_(-float("inf"))
        mask.triu_(1) #set 1 on the diagonal

        return mask
        
    def forward(self, x):
        #loop over the attention blocks
        x = self.token_embedding(x).swapaxes(0,1)
        x = x + self.positional_encoding
        x = self.attention_blocks(x)
        #linear + softmax
        x = self.linear(x.swapaxes(1,2)).squeeze(-1)
        x = self.norm(x)
        return x @ self.proj


class VisionTransformer(nn.Module):
    def __init__(self, embedding_size, num_heads, n_blocks, kernel_size, stride, output_dim, input_resolution, use_mask=False) -> None:
        super().__init__()
        #conv layer to embed patches
        #classical transformer afterward
        #may need 2 layer normalizations (pre and post transformer as in openai)
        
        self.n_blocks = n_blocks
        self.num_heads = num_heads
        scale = embedding_size ** -0.5

        self.positional_encoding = nn.Parameter(scale * torch.randn((input_resolution // kernel_size) ** 2 + 1, embedding_size))
        self.cls = nn.Parameter(scale * torch.randn(embedding_size))

        self.convolution = nn.Conv2d(in_channels=3, out_channels=embedding_size, kernel_size = kernel_size, stride=stride, bias=False)

        self.mask = self.build_mask() if use_mask else None

         # create list of attention block
        attention_blocks = []
        for i in range(n_blocks):
            attention_blocks.append(AttentionBlock(embedding_size=embedding_size, num_heads=num_heads, mask=self.mask))
        self.attention_blocks = nn.Sequential(*attention_blocks)
        
        self.norm = nn.LayerNorm(embedding_size)
        self.proj = nn.Parameter(scale * torch.randn(embedding_size, output_dim))    

    def build_mask(self):
        mask = torch.empty(self.seq_length, self.seq_length)
        mask.fill_(-float("inf"))
        mask.triu_(1) #set 1 on the diagonal

        return mask

    def forward(self, x):
        x = self.convolution(x) #b w grid grid (grid=input_res/kernel_size)
        b, w, grid, grid = x.shape
        x = x.flatten(2).swapaxes(1,2) #b grid**2 w
        cls = self.cls + torch.zeros(b, 1, w) # b 1 w
        x = torch.cat((cls,x), dim=1) # b (input_res/kernel_size) **2 + 1 w
        x = x + self.positional_encoding
        x = self.attention_blocks(x)
        x = self.norm(x[:, 0, :])

        return x @ self.proj


class Clip(nn.Module):
    def __init__(self, embedding_size, vision_embedding, seq_length, num_heads, n_blocks, vocab_size, output_dim, kernel_size, stride, input_resolution) -> None:
        super().__init__()
        #initialize text and image encoders
        #similarity matrix
        self.text_encoder = Transformer(embedding_size=embedding_size,
                                        seq_length=seq_length,
                                        num_heads=num_heads,
                                        n_blocks=n_blocks,
                                        vocab_size=vocab_size,
                                        output_dim=output_dim,
                                        use_mask=True
                                        )


        vision_heads = vision_embedding // 64
        self.image_encoder = VisionTransformer(embedding_size=vision_embedding,
                                               num_heads=vision_heads,
                                               n_blocks=n_blocks,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               output_dim=output_dim, 
                                               input_resolution=input_resolution)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def encode_text(self, x):
        return self.text_encoder(x)
        
    def encode_image(self, x):
        return self.image_encoder(x)

    def forward(self, image, text):
        encoded_image = self.encode_image(image)
        encoded_text = self.encode_text(text)

        #this part is completely from openAI
        # normalized features
        encoded_image = encoded_image / encoded_image.norm(dim=1, keepdim=True)
        encoded_text = encoded_text / encoded_text.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        
        logits_per_image = logit_scale * encoded_image @ encoded_text.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text