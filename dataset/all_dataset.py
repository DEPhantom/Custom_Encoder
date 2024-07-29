import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from ucimlrepo import fetch_ucirepo
import openml
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

class BreastDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        # import dataset
        self.breast_cancer = fetch_ucirepo(id=14)
        self.X = self.breast_cancer.data.features
        self.y = self.breast_cancer.data.targets
        self.transform = transform
        self.target_transform = target_transform
        # preprocess
        label_encoder = preprocessing.LabelEncoder()
        self.X = self.X.fillna("0")
        self.X["age"] = label_encoder.fit_transform( self.X["age"] )
        self.X["menopause"] = label_encoder.fit_transform( self.X["menopause"] )
        self.X["tumor-size"] = label_encoder.fit_transform( self.X["tumor-size"] )
        self.X["inv-nodes"] = label_encoder.fit_transform( self.X["inv-nodes"] )
        self.X["node-caps"] = label_encoder.fit_transform( self.X["node-caps"] )
        self.X["breast"] = label_encoder.fit_transform( self.X["breast"] )
        self.X["breast-quad"] = label_encoder.fit_transform( self.X["breast-quad"] )
        self.X["irradiat"] = label_encoder.fit_transform( self.X["irradiat"] )
        # Normalization
        minmax_scale = preprocessing.MinMaxScaler( feature_range=(0 ,1) )
        self.X = minmax_scale.fit_transform( self.X )
        self.y = label_encoder.fit_transform( self.y )

    def __len__(self):
        return len(self.breast_cancer.data.targets)

    def __getitem__(self, idx):
        # feature = torch.tensor(self.X.iloc[idx])
        feature = torch.tensor(self.X[idx])
        target = torch.tensor(self.y[idx])
        if self.transform:
            pass
        if self.target_transform:
            pass
        return feature, target

    def get_feature_marginal_low(self):
      data =  np.array(self.X)
      return data.min(axis=0)
    ## end get_feature_marginal_low()

    def get_feature_marginal_high(self):
      data =  np.array(self.X)
      return data.max(axis=0)
    ## end get_feature_marginal_high()

# end 

class WineDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        # import dataset
        self.wine = fetch_ucirepo(id=109)
        self.X = self.wine.data.features
        self.y = self.wine.data.targets
        self.transform = transform
        self.target_transform = target_transform
        # preprocess
        label_encoder = preprocessing.LabelEncoder()
        self.X = self.X.fillna("0")
        # Normalization
        minmax_scale = preprocessing.MinMaxScaler( feature_range=(0 ,1) )
        self.X = minmax_scale.fit_transform( self.X )
        self.y = label_encoder.fit_transform( self.y )

    def __len__(self):
        return len(self.wine.data.targets)

    def __getitem__(self, idx):
        # feature = torch.tensor(self.X.iloc[idx])
        feature = torch.tensor(self.X[idx])
        target = torch.tensor(self.y[idx])
        if self.transform:
            pass
        if self.target_transform:
            pass
        return feature, target

    def get_feature_marginal_low(self):
      data =  np.array(self.X)
      return data.min(axis=0)
    ## end get_feature_marginal_low()

    def get_feature_marginal_high(self):
      data =  np.array(self.X)
      return data.max(axis=0)
    ## end get_feature_marginal_high()

# end

class SpambaseDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        # import dataset
        self.spambase = fetch_ucirepo(id=94)
        self.X = self.spambase.data.features
        self.y = self.spambase.data.targets
        self.transform = transform
        self.target_transform = target_transform
        # preprocess
        label_encoder = preprocessing.LabelEncoder()
        self.X = self.X.fillna("0")
        # Normalization
        minmax_scale = preprocessing.MinMaxScaler( feature_range=(0 ,1) )
        self.X = minmax_scale.fit_transform( self.X )
        self.y = label_encoder.fit_transform( self.y )

    def __len__(self):
        return len(self.spambase.data.targets)

    def __getitem__(self, idx):
        feature = torch.tensor(self.X[idx])
        target = torch.tensor(self.y[idx])
        if self.transform:
            pass
        if self.target_transform:
            pass
        return feature, target

    def get_feature_marginal_low(self):
      data =  np.array(self.X)
      return data.min(axis=0)
    ## end get_feature_marginal_low()

    def get_feature_marginal_high(self):
      data =  np.array(self.X)
      return data.max(axis=0)
    ## end get_feature_marginal_high()

# end

class OpenMLDataset(Dataset):
    def __init__(self, task_id, transform=None, target_transform=None, train=None):
        # import dataset
        self.task = openml.tasks.get_task( task_id )
        self.X, self.y = self.task.get_X_and_y()
        self.transform = transform
        self.target_transform = target_transform

        # split data
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        if ( train == None ):
          pass
        elif ( train == True ):
          self.X = X_train
          self.y = y_train
        else:
          self.X = X_test
          self.y = y_test

        # preprocess
        label_encoder = preprocessing.LabelEncoder()
        self.X = np.nan_to_num(self.X, nan=0)
        # Normalization
        minmax_scale = preprocessing.MinMaxScaler( feature_range=(0 ,1) )
        self.X = minmax_scale.fit_transform( self.X )
        self.y = label_encoder.fit_transform( self.y )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        feature = torch.tensor(self.X[idx])
        target = torch.tensor(self.y[idx])
        if self.transform:
            pass
        if self.target_transform:
            pass
        return feature, target

    def get_feature_marginal_low(self):
      data =  np.array(self.X)
      return data.min(axis=0)
    ## end get_feature_marginal_low()

    def get_feature_marginal_high(self):
      data =  np.array(self.X)
      return data.max(axis=0)
    ## end get_feature_marginal_high()

# end

class dataset_info():
    def __init__(self, id, feature_num, feature_dim, class_num, bin_num):
        self.id = id
        self.class_num = class_num
        self.feature_num= feature_num
        self.feature_dim = feature_dim
        self.bin_num = bin_num
        
# end class

class dataset():
    def __init__(self):
        self.data_list = []
        
    def get_list(self, data_type):
        if( data_type == "binary" or data_type == "all" ) :
          self.data_list.append( dataset_info(15, 699, 9, 2, 79 ) )
          self.data_list.append( dataset_info(29, 690, 15, 2, 230 ) )
          self.data_list.append( dataset_info(31, 1000, 20, 2, 148 ) )
          self.data_list.append( dataset_info(37, 768, 8, 2, 269 ) )
          self.data_list.append( dataset_info(49, 958, 9, 2, 19 ) )
          self.data_list.append( dataset_info(3913, 522, 21, 2, 549) )
          self.data_list.append( dataset_info(9952, 5404, 5, 2, 234 ) )
          self.data_list.append( dataset_info(9971, 583, 10, 2, 323) )
          self.data_list.append( dataset_info(10093, 1372, 4, 2, 192 ) )
          self.data_list.append( dataset_info(10101, 748, 4, 2, 82 ) )
          self.data_list.append( dataset_info(146819, 540, 18, 2, 864 ) )
          self.data_list.append( dataset_info(146820, 4839, 5, 2, 240 ) )
        elif( data_type == "multi" or data_type == "all" ) :
          self.data_list.append( dataset_info(11, 625, 4, 3, 16 ) )
          self.data_list.append( dataset_info(18, 2000, 6, 10, 154 ) )
          self.data_list.append( dataset_info(23, 1473, 9, 3, 56 ) )
          self.data_list.append( dataset_info(2079, 736, 19, 5, 318 ) )
          self.data_list.append( dataset_info(3022, 990, 12, 11, 497) )
          self.data_list.append( dataset_info(9960, 5456, 24, 4, 1013 ) )
          self.data_list.append( dataset_info(14969, 9873, 32, 5, 1536 ) )
          self.data_list.append( dataset_info(146817, 1941, 27, 7, 1002 ) )
          self.data_list.append( dataset_info(146821, 1728, 6, 4, 30 ) )
          self.data_list.append( dataset_info(146822, 2310, 16, 7, 670 ) )
        elif( data_type == "high dim" or data_type == "all" ) :
          self.data_list.append( dataset_info(3, 3196, 36, 2, 37 ) )
          self.data_list.append( dataset_info(16, 2000, 64, 10, 3072 ) )
          self.data_list.append( dataset_info(22, 2000, 47, 10, 2256 ) )
          self.data_list.append( dataset_info(43, 4601, 57, 2, 630 ) )
          self.data_list.append( dataset_info(45, 3190, 60, 3, 227 ) )
          self.data_list.append( dataset_info(2074, 6430, 36, 6, 1379 ) )
          self.data_list.append( dataset_info(3481, 7797, 617, 26, 26101) )
          self.data_list.append( dataset_info(3549, 841, 70, 4, 1258) )
          self.data_list.append( dataset_info(3902, 1458, 37, 2, 921) )
          self.data_list.append( dataset_info(3903, 1563, 37, 2, 988) )
          self.data_list.append( dataset_info(3917, 2109, 21, 2, 492) )
          self.data_list.append( dataset_info(3918, 1109, 21, 2, 659) )
          self.data_list.append( dataset_info(9910, 3751, 1776, 2, 6469 ) )
          self.data_list.append( dataset_info(9946, 569, 30, 2, 1434 ) )
          self.data_list.append( dataset_info(9957, 1055, 41, 2, 843 ) )
          self.data_list.append( dataset_info(9976, 2600, 500, 2, 20670 ) )
          self.data_list.append( dataset_info(9978, 2534, 72, 2, 3003) )
          self.data_list.append( dataset_info(9985, 6118, 51, 6, 1649 ) )
          self.data_list.append( dataset_info(125922, 5500, 40, 11, 1920 ) )
          self.data_list.append( dataset_info(146824, 2000, 240, 10, 1415 ) )
        elif( data_type == "large" or data_type == "all" ) :
          self.data_list.append( dataset_info(6, 20000, 16, 26, 174 ) )
          self.data_list.append( dataset_info(32, 10992, 16, 10, 586 ) )
          self.data_list.append( dataset_info(219, 45312, 8, 2, 288 ) )
          self.data_list.append( dataset_info(3904, 10885, 21, 2, 618 ) )
          self.data_list.append( dataset_info(7592, 48842, 14, 2, 178 ) )
          self.data_list.append( dataset_info(9977, 34465, 118, 2, 997 ) )
          self.data_list.append( dataset_info(14952, 11055, 30, 25, 38 ) )
          self.data_list.append( dataset_info(14965, 45211, 16, 2, 214 ) )
          self.data_list.append( dataset_info(14970, 10299, 561, 6, 26284 ) )
          self.data_list.append( dataset_info(167119, 44819, 6, 3, 36 ) )
          self.data_list.append( dataset_info(167120, 96320, 21, 2, 1008 ) )
        else :
          pass
        
        return self.data_list

# end
