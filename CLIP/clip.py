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
    def __init__(self, embedding_size, seq_length, num_heads, n_blocks, vocab_size) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_blocks = n_blocks
        self.num_heads = num_heads

        self.positional_encoding = nn.Parameter(torch.empty(seq_length, embedding_size))
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)

        # create list of attention block
        attention_blocks = []
        for i in range(n_blocks):
            attention_blocks.append(AttentionBlock(embedding_size=embedding_size, num_heads=num_heads))
        self.attention_blocks = nn.Sequential(*attention_blocks)

        #may be needed: layer norm, linear from encoded vals to probs
        
    def forward(self, x):
        #loop over the attention blocks
        x = self.token_embedding(x)
        x = x + self.positional_encoding(x)

        x = self.attenion_blocks(x)
        #linear + softmax
        return x


class VisionTransformer(nn.Module):
    def __init__(self, embedding_size, num_heads, n_blocks, kernel_size, stride, output_dim, input_resolution) -> None:
        super().__init__()
        #conv layer to embed patches
        #classical transformer afterward
        #may need 2 layer normalizations (pre and post transformer as in openai)
        
        self.n_blocks = n_blocks
        self.num_heads = num_heads

        scale = embedding_size ** -0.5
        self.positional_encoding = nn.Parameter(torch.empty(scale * torch.randn((input_resolution // kernel_size) ** 2 + 1, embedding_size)))
        self.class_embedding = nn.Parameter(scale * torch.randn(embedding_size))

        self.convolution = nn.Conv2d(in_channels=3, out_channels=embedding_size, kernel_size = kernel_size, stride=stride, bias=False)

         # create list of attention block
        attention_blocks = []
        for i in range(n_blocks):
            attention_blocks.append(AttentionBlock(embedding_size=embedding_size, num_heads=num_heads))
        self.attention_blocks = nn.Sequential(*attention_blocks)

        
        self.proj = nn.Parameter(scale * torch.randn(embedding_size, output_dim))

    def forward(self, x):
        x = self.convolution(x)
        x = x + self.positional_encoding(x)

        x = self.attention_blocks(x)

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
                                        vocab_size=vocab_size
                                        )

        vision_heads = vision_embedding // 64

        self.image_encoder = VisionTransformer(embedding_size=vision_embedding,
                                               num_heads=num_heads,
                                               n_blocks=n_blocks,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               output_dim=output_dim, 
                                               input_resolution=input_resolution)

        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(1 / 0.07))

    def encode_text(self, x):
        return self.text_encoder(x)
        
    def encode_image(self, x):
        return self.image_encoder(x)

    def forward(self, image, text):
        encoded_image = self.encode_image(image)
        encoded_text = self.encode_text(text)


        #this part is completely from openAI
        # normalized features
        encoded = encoded / encoded_image.norm(dim=1, keepdim=True)
        encoded_text = encoded_text / encoded_text.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * encoded_image @ encoded_text.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text