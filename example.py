import torch
import torch.nn as nn
from module import scarf_exp

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

def main():
  exp = scarf_exp.SCARF_Experiment(standard_code, "large", encoder_reshape=True )
  exp.run()

if __name__ == '__main__':
  main()


