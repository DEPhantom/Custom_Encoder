import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
from .Custom_Encoder import CE_Module
import math
from . import rtdl
from typing import List

class MLP(torch.nn.Sequential):
    def __init__(self, input_dim, hidden_dim, num_hidden, dropout=0.0):
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.extend([nn.Linear(in_dim, hidden_dim), nn.Dropout(dropout)])

        super().__init__(*layers)

class _Periodic(nn.Module):

    def __init__(self, n_features: int, k: int, sigma: float) -> None:
        if sigma <= 0.0:
            raise ValueError(f'sigma must be positive, however: {sigma=}')

        super().__init__()
        self._sigma = sigma
        self.weight = nn.Parameter(torch.empty(n_features, k))
        self.reset_parameters()
        self.k = k

    def reset_parameters(self):
        # NOTE[DIFF]
        # Here, extreme values (~0.3% probability) are explicitly avoided just in case.
        # In the paper, there was no protection from extreme values.
        bound = self._sigma * 3
        nn.init.trunc_normal_(self.weight, 0.0, self._sigma, a=-bound, b=bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )

        x = 2 * math.pi * self.weight * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x
        
    def get_encoder_dim(self, input_dim):
        output_dim = input_dim*2*self.k
        return output_dim

class standard_code(nn.Module):

    def __init__(self, input_x ) -> None:
        super().__init__()
        # calculate mean
        self.mean = torch.mean(input_x, 0)
        # calculate sigma
        self.sigma = torch.std(input_x, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for i, sigma in enumerate(self.sigma):
          row = x.t()[i].clone()
          row_mean = row.clone()
          row_sigma = row.clone()
          row_mean[row_mean>self.mean[i]] = 1
          row_mean[row_mean<self.mean[i]] = 0
          row_sigma[row_sigma>self.sigma[i]+self.mean[i]] = 1
          row_sigma[row_sigma<self.sigma[i]-self.mean[i]] = 1
          row_sigma[row_sigma<self.sigma[i]+self.mean[i]] = 0
          row_sigma[row_sigma>self.sigma[i]-self.mean[i]] = 0
          encode = torch.stack((row_mean,row_sigma),1)
          if ( i == 0 ):
            output = encode
          else:
            output = torch.cat( ( output, encode ), 1 )

        return output
        
    def get_encoder_dim(self, input_dim):
        output_dim = input_dim*2
        return output_dim

class SCARF(CE_Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        features_low,
        features_high,
        num_hidden=4,
        head_depth=2,
        corruption_rate=0.6,
        dropout=0.0,
    ):
        super().__init__()

        self.encoder = MLP(input_dim, emb_dim, num_hidden, dropout)
        self.pretraining_head = MLP(emb_dim, emb_dim, head_depth)

        # uniform distribution over marginal distributions of dataset's features
        self.marginals = Uniform(torch.Tensor(features_low), torch.Tensor(features_high))
        self.corruption_len = int(corruption_rate * input_dim)

    def forward(self, x):
        batch_size, m = x.size()

        # 1: create a mask of size (batch size, m) where for each sample we set the jth column to True at random, such that corruption_len / m = corruption_rate
        # 2: create a random tensor of size (batch size, m) drawn from the uniform distribution defined by the min, max values of the training set
        # 3: replace x_corrupted_ij by x_random_ij where mask_ij is true

        corruption_mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        for i in range(batch_size):
            corruption_idx = torch.randperm(m)[: self.corruption_len]
            corruption_mask[i, corruption_idx] = True

        x_random = self.marginals.sample(torch.Size((batch_size,))).to(x.device)
        x_corrupted = torch.where(corruption_mask, x_random, x)

        # Custom Encoder
        x = self.forward_encoder(x)
        x_corrupted = self.forward_encoder(x_corrupted)

        embeddings = self.encoder(x)
        embeddings = self.pretraining_head(embeddings)

        embeddings_corrupted = self.encoder(x_corrupted)
        embeddings_corrupted = self.pretraining_head(embeddings_corrupted)

        return embeddings, embeddings_corrupted
    
    def adjust_structure(self):
      self.encoder[0] = self.adjust_layer( self.encoder[0] )

    def get_embedding(self):
        return self.encoder

    @torch.inference_mode()
    def get_embeddings(self, x):
        return self.encoder(x)

class finetune_model(nn.Module):
    def __init__(self, input_dim, emb_dim, class_num, encoder, name="None", personalized_encode=None, reshape_dim=0, num_hidden=4, corruption_rate=0.6, dropout=0.0,
                 n_frequencies=8, frequency_init_scale: float = 0.01,):
        super(finetune_model, self).__init__()
        self.classification_num = class_num
        # Network structure

        self.personalized_encode = personalized_encode

        self.encoder = encoder
        self.name = name
        self.reshape_dim = reshape_dim
        self.classification_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, self.classification_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        batch_size, m = x.size()
        if ( self.name != "None" ) :
          x = self.personalized_encode(x)

        if ( self.reshape_dim != 0 ):
          x = torch.reshape(x, (batch_size, self.reshape_dim))

        h = self.encoder(x)
        c = self.classification_head(h)
        # c = torch.argmax(c, dim=1)
        return c

    def forward_classifier(self, x):
        x = x.to(torch.float32)
        batch_size, m = x.size()
        if ( self.name != "None" ) :
          x = self.personalized_encode(x)

        if ( self.reshape_dim != 0 ):
          x = torch.reshape(x, (batch_size, self.reshape_dim))

        h = self.encoder(x)
        c = self.classification_head(h)
        c = torch.argmax(c, dim=1)
        return c

# end
