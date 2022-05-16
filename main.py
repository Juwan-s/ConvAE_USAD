import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import torch
import torch.nn as nn
import torch.utils.data as data_utils
pd.set_option('display.max_columns', None)

from sklearn import preprocessing
from sklearn.metrics import roc_curve,roc_auc_score, f1_score, classification_report, confusion_matrix
from numpy import argmax
from utils import *
from usad import *

import argparse

device = get_default_device()


torch.manual_seed(42)


def downsample(df:pd.DataFrame, label_col_name:str) -> pd.DataFrame:
    # find the number of observations in the smallest group
    nmin = df[label_col_name].value_counts().min()
    return (df
            # split the dataframe per group
            .groupby(label_col_name)
            # sample nmin observations from each group
            .apply(lambda x: x.sample(nmin))
            # recombine the dataframes
            .reset_index(drop=True)
            )

parser = argparse.ArgumentParser(description='python main.py')

parser.add_argument('--dataset', default='swat', help='swat or wadi')
parser.add_argument('--window_size', default=10, help='window_size')
parser.add_argument('--latent_size', default=10, help='latent_size')
parser.add_argument('--batch_size', default=4096, help='batch_size')
parser.add_argument('--epoch', default=70, help='epoch')
parser.add_argument('--alpha', default=0.5, help='alpha')
parser.add_argument('--model', required = True, choices = ['conv1d', 'linear'],help='Please set Model Name : \'conv1d\' or \'linear\'')
parser.add_argument('--undersample', required = True, choices = ['yes', 'no'],help='Please set Under Sample action : \'yes\' or \'no\'')

args = parser.parse_args()




print("Loading Datasets")

normal_data = args.dataset + "_normal_preprocessed.csv"
attack_data = args.dataset + "_attack_preprocessed.csv"

labels = args.dataset + "_labels.csv"

normal = pd.read_csv(normal_data)

attack = pd.read_csv(attack_data)

labels_df = pd.read_csv(labels)

labels = list(labels_df['0'])

print("Dataset Loaded....")

window_size = int(args.window_size)

BATCH_SIZE =  int(args.batch_size)
N_EPOCHS = int(args.epoch)
hidden_size = int(args.latent_size)

print("Creating Windows..")
windows_normal=normal.values[np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None]]

windows_attack=attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None]]

if args.model == 'conv1d':
    print("Training 1d Conv AutoEncoder")
    w_size = windows_normal.shape[1]

    # k_size = w_size // 2

    if w_size >= 30:
        k_size = w_size // 3
    elif w_size >= 20:
        k_size = w_size // 3
    else:
        k_size = w_size // 2

    if args.dataset == "swat":
        config = {
            'first_layer' : 64,
            'feature_dim' : 51,
        }
    else:
        config = {
            'first_layer' : 128,
            'feature_dim' : 123,
        }



    z_size = hidden_size

    windows_normal_train = windows_normal[:int(np.floor(.8 *  windows_normal.shape[0]))]
    windows_normal_val = windows_normal[int(np.floor(.8 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

    # train, validation split
    print("windows created")

    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal_train).float().permute(0,2,1).contiguous()
    ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal_val).float().permute(0,2,1).contiguous()
    ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_attack).float().permute(0,2,1).contiguous()
    ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = Conv1dModel(w_size, z_size, k_size, config)
    model = to_device(model,device)

    print("Model is Loaded. Start Trainig")

    print("Training Config")

    config_message = "Epoch : {0} | Latent_size : {1} | Alpha : {2} | Window size : {3} | Batch size : {4} | model : {5} | dataset : {6}".format(N_EPOCHS,hidden_size, float(args.alpha), window_size, BATCH_SIZE, args.model, args.dataset)

    print(config_message)

    history = Conv1dtraining(N_EPOCHS,model,train_loader,val_loader)

    print("Start Testing")
    results=Conv1dtesting(model,test_loader, alpha = float(args.alpha), beta = 1 - float(args.alpha))

    windows_labels=[]
    for i in range(len(labels)-window_size):
        windows_labels.append(list(np.int_(labels[i:i+window_size])))

    y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]

    y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                results[-1].flatten().detach().cpu().numpy()])

    print("Find threshold")
    fpr,tpr,tr=roc_curve(y_test,y_pred)
    J = tpr - fpr
    ix = argmax(J)
    best_thresh = tr[ix]

    y_pred_label = [1.0 if (score > best_thresh) else 0 for score in y_pred ]


    if args.undersample == "yes":
        print("Starting Undersampling...")
        y_test_df = pd.DataFrame(y_test)
        y_pred_df = pd.DataFrame(y_pred_label)



        confusion_df = pd.concat([y_pred_df, y_test_df], axis = 1)

        confusion_df.columns = ["pred", "label"]

        balanced = downsample(confusion_df, "label")
        y_pred_label = list(balanced["pred"])
        y_test = list(balanced['label'])


    print("========================================")
    print(classification_report(y_test, y_pred_label))

else:

    print("Training Linear AutoEncoder")

    w_size=windows_normal.shape[1]*windows_normal.shape[2]
    z_size=windows_normal.shape[1]*hidden_size

    windows_normal_train = windows_normal[:int(np.floor(.8 *  windows_normal.shape[0]))]
    windows_normal_val = windows_normal[int(np.floor(.8 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]
    
    print("windows created")

    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],w_size]))
    ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],w_size]))
    ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0],w_size]))
    ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UsadModel(w_size, z_size)
    model = to_device(model,device)

    print("Model is Loaded. Start Trainig")

    print("Training Config")

    config_message = "Epoch : {0} | Latent_size : {1} | Alpha : {2} | Window size : {3} | Batch size : {4} | model : {5} | dataset : {6}".format(N_EPOCHS,hidden_size, float(args.alpha), window_size, BATCH_SIZE, args.model, args.dataset)

    print(config_message)


    history = Usadtraining(N_EPOCHS,model,train_loader,val_loader)

    print("Start Testing")

    results=Usadtesting(model,test_loader)
    windows_labels=[]
    for i in range(len(labels)-window_size):
        windows_labels.append(list(np.int_(labels[i:i+window_size])))
    y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels ]
    y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                              results[-1].flatten().detach().cpu().numpy()])


    print("Find threshold")

    fpr,tpr,tr=roc_curve(y_test,y_pred)
    J = tpr - fpr
    ix = argmax(J)
    best_thresh = tr[ix]
    y_pred_label = [1.0 if (score > best_thresh) else 0 for score in y_pred ]

    if args.undersample == "yes":
        
        print("Starting Undersampling...")
        y_test_df = pd.DataFrame(y_test)
        y_pred_df = pd.DataFrame(y_pred_label)



        confusion_df = pd.concat([y_pred_df, y_test_df], axis = 1)

        confusion_df.columns = ["pred", "label"]

        balanced = downsample(confusion_df, "label")
        y_pred_label = list(balanced["pred"])
        y_test = list(balanced['label'])


    print("========================================")

    print(classification_report(y_test, y_pred_label))