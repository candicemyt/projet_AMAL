import torch 
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.activation import NonDynamicallyQuantizableLinear
import numpy as np
import math


class Attentionlayer(nn.Module):
    def __init__(self, embedding_size, num_heads, bias=True, dropout_p=0.) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.embedding_size = embedding_size

        self.in_linear = nn.Linear(embedding_size, embedding_size*3, bias=bias)
        #self.out_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.out_linear = NonDynamicallyQuantizableLinear(embedding_size, embedding_size, bias=bias)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, q, k, v,  mask):
        l, b, d = q.shape #length, batch size, embedding dim
        head_dim = d // self.num_heads
        assert head_dim * self.num_heads == self.embedding_size, "embedding_size must be divisible by num_heads"
        if q.equal(k) and k.equal(v):
            #in projection (to get q, k, v):
            q, k, v = self.in_linear(q).chunk(3, dim=-1)  
        else:
            assert False, "Different Query Key Value not implemented"

        #in openai clip no k bias or v bias were added

        #Rearrange q,k,v d/num_head        
        q = q.contiguous().view(l, b * self.num_heads, head_dim).transpose(0,1)
        if k is not None:
            k = k.contiguous().view(-1, b * self.num_heads, head_dim).transpose(0,1)
        if v is not None:
            v = v.contiguous().view(-1, b * self.num_heads, head_dim).transpose(0,1)

        #in openai clip no zero attention padding
        _, _, d_q = q.shape #embedding dim after rearrange

        #compute attention
        q = q / math.sqrt(d_q)
        if mask is not None:
            z = torch.bmm(q, k.transpose(-2,-1)) + mask
        else:
            z = torch.bmm(q, k.transpose(-2,-1))
        z = F.softmax(z, dim=-1)
        #Dropout (in CLIP dorpout p is set to 0, so this line is +/- useless)
        z = self.dropout(z)
        z = torch.bmm(z,v)

        z = z.transpose(0, 1).contiguous().view(l * b, self.embedding_size) #as in pytorch
        #out projection (i.e concatenation of different heads)
        z = self.out_linear(z)
        return z.view(l, b, z.size(1))



class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class AttentionBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, mask=None) -> None:
        super().__init__()
        self.attention_layer = Attentionlayer(embedding_size=embedding_size, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            QuickGELU(),
            nn.Linear(embedding_size * 4, embedding_size)
        )
        self.norm2 = nn.LayerNorm(embedding_size)
        self.mask = mask

    def attention(self, x):
        self.mask = self.mask.to(dtype=x.dtype, device=x.device) if self.mask is not None else None
        return self.attention_layer(x, x, x, mask=self.mask)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self,
                 embedding_size,
                 seq_length,
                 num_heads,
                 n_blocks,
                 vocab_size,
                 output_dim,
                 use_mask=True) -> None:

        super().__init__()
        self.vocab_size = vocab_size
        self.n_blocks = n_blocks
        self.num_heads = num_heads
        self.seq_length = seq_length

        self.mask = self.build_mask() if use_mask else None

        self.positional_encoding = nn.Parameter(torch.empty(seq_length, embedding_size))
        self.token_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        # create list of attention block
        attention_blocks = [AttentionBlock(embedding_size=embedding_size, num_heads=num_heads, mask=self.mask) 
                            for _ in range(n_blocks)]
        self.attention_blocks = nn.Sequential(*attention_blocks)

        self.norm = nn.LayerNorm(embedding_size)
        self.proj = nn.Parameter(torch.rand(embedding_size, output_dim))

    def build_mask(self):
        mask = torch.empty(self.seq_length, self.seq_length)
        mask.fill_(-float("inf"))
        mask.triu_(1) #set 1 on the diagonal
        return mask
        
    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_encoding
        x = x.permute(1, 0, 2) # BLD -> LBD
        x = self.attention_blocks(x)
        x = x.permute(1, 0, 2) #LBD ->BLD
        x = self.norm(x)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.proj

        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 embedding_size,
                 num_heads,
                 n_blocks,
                 kernel_size,
                 stride,
                 output_dim,
                 input_resolution,
                 use_mask=False) -> None:

        super().__init__()
        self.n_blocks = n_blocks
        self.num_heads = num_heads
        scale = embedding_size ** -0.5

        self.positional_encoding = nn.Parameter(scale * torch.randn((input_resolution // kernel_size) ** 2 + 1, embedding_size))
        self.cls = nn.Parameter(scale * torch.randn(embedding_size))

        self.convolution = nn.Conv2d(in_channels=3, out_channels=embedding_size, kernel_size = kernel_size, stride=stride, bias=False)

        # create list of attention block
        attention_blocks = [AttentionBlock(embedding_size=embedding_size, num_heads=num_heads, mask=self.mask) 
                            for _ in range(n_blocks)]
        self.attention_blocks = nn.Sequential(*attention_blocks)
        
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.proj = nn.Parameter(scale * torch.randn(embedding_size, output_dim))    

    def forward(self, x):
        x = self.convolution(x) #b w grid grid (grid=input_res/kernel_size)
        b, w, grid, grid = x.shape
        x = x.flatten(2).swapaxes(1,2) #b grid**2 w
        cls = self.cls + torch.zeros(b, 1, w) # b 1 w
        x = torch.cat((cls,x), dim=1) # b (input_res/kernel_size) **2 + 1 w
        x = x + self.positional_encoding
        x = self.norm1(x)
        x = x.permute(1, 0, 2) #BLD -> LBD
        x = self.attention_blocks(x)
        x = x.permute(1, 0, 2) #LBD -> BLD
        x = self.norm2(x[:, 0, :]) #normalize the first element of the seq

        return x @ self.proj


class Clip(nn.Module):
    def __init__(self,
                 embedding_size,
                 vision_embedding,
                 seq_length,
                 num_heads,
                 n_blocks,
                 vocab_size,
                 output_dim,
                 kernel_size,
                 stride,
                 input_resolution) -> None:

        super().__init__()
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