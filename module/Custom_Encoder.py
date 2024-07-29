import torch
import torch.nn as nn

class CE_Module(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.personalized_embedding = None
        self.encoder_dim = 0
        self.reshape = False

    def forward_encoder(self, x):
        
        batch_size, m = x.size()
        
        # get embeddings
        if ( self.encoder_dim == 0 ) :
          return x
        else :
          x = self.personalized_embedding(x)
        # end else

        if ( self.reshape == True ) :
          x = torch.reshape(x, (batch_size, self.encoder_dim))
        # end if

        return x

    def adjust_structure(self):
        pass

    def set_encoder(self, callback=None, encoder_dim=0, reshape=False ) :
        self.personalized_embedding = callback
        self.encoder_dim = encoder_dim
        self.reshape = reshape
        self.adjust_structure()

    # end set_encoder

    def adjust_layer( self, layer ):
       out_dim = layer.out_features
       return nn.Linear( self.encoder_dim, out_dim )

    def get_encoder_dim(self) :
        return self.encoder_dim

    def check_encoder_set(self) :
        if (self.encoder_dim == 0 ):
          return False
        else :
          return True

    # end def
