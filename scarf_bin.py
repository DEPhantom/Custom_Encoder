import os
import argparse
import torch
import numpy as np
from utils import yaml_config_hook, save_model
from scarf import scarf_model, scarf_loss, rtdl
from dataset import all_dataset
from sklearn import metrics
from munkres import Munkres

def inference(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c = model.forward_cluster(x)
        c = c.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

def evaluate(label, pred):
    acc = metrics.accuracy_score(pred, label)
    precision = metrics.precision_score(pred, label,average='macro')
    recall = metrics.recall_score(pred, label,average='macro')
    return acc, precision, recall

def main():
  # -------------------- Define Param -------------------------------------------
  parser = argparse.ArgumentParser()
  config = yaml_config_hook("./config/config.yaml")
  for k, v in config.items():
    parser.add_argument(f"--{k}", default=v, type=type(v))
    
  args = parser.parse_args()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # -----------------------------------------------------------------------------

  # -------------------- Read Data ----------------------------------------------
  if args.dataset == "Breast":
    train_dataset = all_dataset.BreastDataset()
    # dataset = data.ConcatDataset([train_dataset, test_dataset])
    dataset = train_dataset
    class_num = 2
  elif args.dataset == "Wine":
    train_dataset = all_dataset.WineDataset()
    # dataset = data.ConcatDataset([train_dataset, test_dataset])
    dataset = train_dataset
    class_num = 3
  elif args.dataset == "Spambase":
    train_dataset = all_dataset.SpambaseDataset()
    # dataset = data.ConcatDataset([train_dataset, test_dataset])
    dataset = train_dataset
    class_num = 2
  elif args.dataset == "OpenML":
    train_dataset = all_dataset.OpenMLDataset(int(args.dataset_id), train=True)
    full_dataset = all_dataset.OpenMLDataset(int(args.dataset_id))
    dataset = train_dataset
    class_num = args.dataset_class
  else:
    raise NotImplementedError
        
  data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.workers,
  )
  
  # -----------------------------------------------------------------------------
  
  # -------------------- Create Network -----------------------------------------
  all_data = full_dataset[:] 
  features, labels = all_data
  bins = bins = rtdl.compute_bins(features)

  model = scarf_model.SCARF_bin(
    input_dim=args.feature_dim,
    bin_dim=args.bin_dim,
    emb_dim=args.emb_dim,
    features_low=dataset.get_feature_marginal_low(),
    features_high=dataset.get_feature_marginal_high(),
    bins=bins
  ).to(device)
  
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
  
  criterion = scarf_loss.InfoNCE()
  # -----------------------------------------------------------------------------
    
  # -------------------- Training -----------------------------------------------
  for epoch in range(args.start_epoch, args.epochs):
    lr = optimizer.param_groups[0]["lr"]
    # loss
    loss_epoch = 0
    for step, (x, y) in enumerate(data_loader): 
        # zero the parameter gradients
        optimizer.zero_grad()
        x = x.to("cuda")
        # forward + backward + optimize
        emb_anchor, emb_positive = model(x)
        loss = criterion(emb_anchor, emb_positive)
        loss.backward()
        optimizer.step()
        
        loss_epoch += loss.item()
        
    # end for    
        
    if epoch % 100 == 0:
        """
        X, Y = inference(data_loader, model, device)
        acc, precision, recall = evaluate(Y, X)
        print('Accuracy = {:.4f} Precision = {:.4f} recall = {:.4f}'.format(acc>
        """
        pass
    # end if

    print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
   
  # end for
  
  save_model.save_model(args, model.get_embedding())
  # save_model_checkpoint(args, model, optimizer, args.epochs)
  # -----------------------------------------------------------------------------
  
  print( "Finished SCARF bin" )

## end main()

if __name__ == '__main__':
  main()
