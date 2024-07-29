import argparse
import torch
import numpy as np
from utils import yaml_config_hook
from . import scarf_model, scarf_loss, contrastive_loss, rtdl
from dataset import all_dataset
from sklearn import metrics
from munkres import Munkres

class SCARF_Experiment():

    def __init__(self, custom_encoder_cls = None, dataset_type = "all", encoder_reshape = False ) -> None:
        parser = argparse.ArgumentParser()
        config = yaml_config_hook("./config/config.yaml")
        for k, v in config.items():
          parser.add_argument(f"--{k}", default=v, type=type(v))
    
        self.args = parser.parse_args()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = None
        self.train_dataset = None
        self.full_dataset = None
        self.test_dataset = None
        self.class_num = 0
        self.callback = None
        self.encoder_dim = 0
        self.encoder_reshape = encoder_reshape
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.finetune_model = None
        self.custom_encoder_cls = custom_encoder_cls
        self.data_dict = all_dataset.dataset().get_list(dataset_type) # all, binary, multi, high dim, large
        self.exp_count = 0

    def inference(self, loader, model, device):
        model.eval()
        feature_vector = []
        labels_vector = []
        for step, (x, y) in enumerate(loader):
            x = x.to(device)
            with torch.no_grad():
                c = model.forward_classifier(x)
            c = c.detach()
            feature_vector.extend(c.cpu().detach().numpy())
            labels_vector.extend(y.numpy())
            if step % 20 == 0:
                print(f"Step [{step}/{len(loader)}]\t Computing features...")
        feature_vector = np.array(feature_vector)
        labels_vector = np.array(labels_vector)
        print("Features shape {}".format(feature_vector.shape))
        return feature_vector, labels_vector

    def evaluate(self, label, pred):
        acc = metrics.accuracy_score(pred, label)
        precision = metrics.precision_score(pred, label,average='macro')
        recall = metrics.recall_score(pred, label,average='macro')
        return acc, precision, recall


    def load_data(self, id):
        args = self.args
        self.train_dataset = all_dataset.OpenMLDataset(int(self.data_dict[self.exp_count].id), train=True)
        self.full_dataset = all_dataset.OpenMLDataset(int(self.data_dict[self.exp_count].id))
        self.test_dataset = all_dataset.OpenMLDataset(int(self.data_dict[self.exp_count].id), train=False)
        self.class_num = self.data_dict[self.exp_count].class_num

    def reset_model(self):
        args = self.args
        
        all_data = self.full_dataset[:]
        features, labels = all_data
        
        # -------------------- Create Custom Encoder -----------------------------------------
        if ( self.args.encoder == "None" ):
          self.callback = None
          self.encoder_dim = 0
          self.encoder_reshape = False
        elif ( self.args.encoder == "Periodic" ):
          self.callback = _Periodic(self.args.feature_dim, n_frequencies, frequency_init_scale)
          self.encoder_dim = self.callback.get_encoder_dim(self.data_dict[self.exp_count].feature_dim)
          self.encoder_reshape = True
        elif ( self.args.encoder == "PieceWise" ):
          bins = rtdl.compute_bins(features)
          self.callback = rtdl.PiecewiseLinearEncoding(bins)
          self.encoder_dim = self.data_dict[self.exp_count].bin_num
          self.encoder_reshape = False
        elif ( self.args.encoder == "std" ):
          self.callback = scarf_model.standard_code(features)
          self.encoder_dim = self.callback.get_encoder_dim(self.data_dict[self.exp_count].feature_dim)
          self.encoder_reshape = True
        else:
          # Custom
          self.callback = self.custom_encoder_cls(features)
          self.encoder_dim = self.callback.get_encoder_dim(self.data_dict[self.exp_count].feature_dim)
        
        # -----------------------------------------------------------------------------
        
        # -------------------- Create Network -----------------------------------------
        self.model = scarf_model.SCARF(
            input_dim=self.data_dict[self.exp_count].feature_dim,
            emb_dim=args.emb_dim,
            features_low=self.train_dataset.get_feature_marginal_low(),
            features_high=self.train_dataset.get_feature_marginal_high(),
        )

        self.model.set_encoder(callback=self.callback, encoder_dim=self.encoder_dim, reshape=self.encoder_reshape )
        self.model.to(self.device)
       
  
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
  
        self.criterion = scarf_loss.InfoNCE()


    def train(self):
        args = self.args
        self.dataset = self.train_dataset
        data_loader = torch.utils.data.DataLoader(
          self.dataset,
          batch_size=args.batch_size,
          shuffle=True,
          drop_last=True,
          num_workers=args.workers,
        )
    
    
        for epoch in range(args.start_epoch, args.epochs):
          lr = self.optimizer.param_groups[0]["lr"]
          # loss
          loss_epoch = 0
          for step, (x, y) in enumerate(data_loader): 
            # zero the parameter gradients
            self.optimizer.zero_grad()
            x = x.to("cuda")
            # forward + backward + optimize
            emb_anchor, emb_positive = self.model(x)
            loss = self.criterion(emb_anchor, emb_positive)
            loss.backward()
            self.optimizer.step()
        
            loss_epoch += loss.item()
        
          # end for    

          print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
   
        # end for

    # end train

    def finetune(self) :
        args = self.args
        self.dataset = self.test_dataset
        if ( self.data_dict[self.exp_count].feature_num/10 < args.batch_size ) :
          batch_size = self.data_dict[self.exp_count].feature_num//10
        else:
          batch_size=args.batch_size
          
        data_loader = torch.utils.data.DataLoader(
          self.dataset,
          batch_size=batch_size,
          shuffle=True,
          drop_last=True,
          num_workers=args.workers,
        )
    
        # -------------------- Create Network -----------------------------------------
        encoder = self.model.get_embedding()
        if ( args.encoder == "None" ) :
            reshape_dim=0
        else :
            if ( args.encoder == "std" ):
                reshape_dim=self.callback.get_encoder_dim(self.data_dict[self.exp_count].feature_dim)
            elif ( args.encoder == "Periodic" ):
                reshape_dim=self.callback.get_encoder_dim(self.data_dict[self.exp_count].feature_dim)
            else :
                # custom
                if ( self.encoder_reshape == True ):
                  reshape_dim=self.callback.get_encoder_dim(self.data_dict[self.exp_count].feature_dim)
                else:
                  reshape_dim=0


        self.finetune_model = scarf_model.finetune_model(
          input_dim=self.data_dict[self.exp_count].feature_dim,
          emb_dim=args.emb_dim,
          class_num=self.data_dict[self.exp_count].class_num,
          encoder=encoder,
          name=args.encoder,
          personalized_encode=self.callback,
          reshape_dim=reshape_dim
        ).to(self.device)

        optimizer = torch.optim.Adam(self.finetune_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        criterion_classification = torch.nn.CrossEntropyLoss()
        # -----------------------------------------------------------------------------
  
        # -------------------- Finetuning -----------------------------------------------
        for epoch in range(args.start_epoch, args.finetune_epochs):
          lr = optimizer.param_groups[0]["lr"]
          # loss
          loss_epoch = 0
          for step, (x, y) in enumerate(data_loader):
            # zero the parameter gradients
            optimizer.zero_grad()
            x = x.to("cuda")
            y = y.to("cuda")
            # forward + backward + optimize
            c = self.finetune_model(x)

            # loss = criterion_classification( c.float(), y.float() )
            loss = criterion_classification( c, y )
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()

          # end for
    
          print(f"Epoch [{epoch}/{args.finetune_epochs}]\t Loss: {loss_epoch / len(data_loader)}")

        # end for
        
    def predict(self) :
        args = self.args
        
        if ( self.data_dict[self.exp_count].feature_num/10 < args.batch_size ) :
          batch_size = self.data_dict[self.exp_count].feature_num//10
        else:
          batch_size=args.batch_size
          
        # -----------------------------------------------------------------------------
        data_loader = torch.utils.data.DataLoader(
          self.test_dataset,
          batch_size=batch_size,
          shuffle=True,
          drop_last=True,
          num_workers=args.workers,
        )

        # -------------------- Testing ------------------------------------------------
        X, Y = self.inference(data_loader, self.finetune_model, self.device)
        acc, precision, recall = self.evaluate(Y, X)
        print('Accuracy = {:.4f} Precision = {:.4f} recall = {:.4f}'.format(acc, precision, recall))
        # -----------------------------------------------------------------------------
    
    def run(self):
        while( self.exp_count < len(self.data_dict) ):
            self.load_data( self.exp_count )
            self.reset_model()
            self.train()
            self.finetune()
            self.predict()
            self.exp_count+= 1
        # end while
        
    # end run
        
