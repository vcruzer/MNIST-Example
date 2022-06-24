import os, sys, pickle
import sklearn, torch
import xgboost as xgb
import keras_tuner as kt
import tensorflow as tf
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from ray import tune
from multiprocessing import cpu_count
from utils_dataset import read_mnist
from mnist_xgb import flatten
from sklearn.model_selection import train_test_split
import utils_hp


num_trials = 1#20
EPOCHS = 2 #10
BATCH_SIZE = 128

#k-fold search using sklearn
def xgb_explore(data_x,data_y):
    pass
def tf_explore(data_x,data_y):
    pass

#random search with RayTune
def torch_explore(data_x,data_y):

    def train_mnist(config,data):

        train_x, val_x, train_y, val_y = train_test_split(data[0],data[1], test_size=0.2, shuffle=True)
        val_x = torch.tensor(val_x)

        train_dset = []
        for i in range(len(train_y)):
            train_dset.append((train_x[i],train_y[i]))

        trainloader = torch.utils.data.DataLoader(train_dset, batch_size=BATCH_SIZE, num_workers=4)
        model = utils_hp.HPConvNet(config,10)

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

        patience = 3
        patience_counter = 0; 
        for epoch in range(EPOCHS):
            #batch training
            for batch_id, (batch_x, batch_y) in enumerate(trainloader):

                ypred = model(batch_x)

                loss = F.cross_entropy(ypred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #scheduler.step()
            

            #validation step
            with torch.no_grad():
                ypred = model(val_x)
                ypred = torch.argmax(ypred,dim=1).detach().numpy()

            val_acc = accuracy_score(val_y,ypred)
            print(f'Validation Acc: {val_acc:.4f}')
            #tune.report(loss=loss.item(), accuracy=val_acc) #!reporting back to raytune
            
            #Early stopping + rollback
            if(epoch==0):
                previous_step_acc = val_acc
            else:

                patience_counter+=1
                if(val_acc <= previous_step_acc) and (patience_counter==patience):
                    break
                elif(val_acc > previous_step_acc):
                    previous_step_acc = val_acc
                    patience_counter = 0
        


    #filling radom search
    config = {
            "cnn_0": 32,
            "dropout_0": 0.5,
            "num_conv": 2,
            "num_dense": 0,

            "lr": 0.1,
            #"batch_size": tune.choice([16,32,64,128]),
            #"epochs": tune.grid_search([5,10,15]),
            
        }
    
    for i in range(1,4):
        config[f'dense_{i}'] = 16
        config[f'dropout_{i}'] = 0.5
        config[f'cnn_{i}'] = 32


    data_x = data_x.astype("float32") / 255
    data_x = np.expand_dims(data_x, 1)
    print(data_x.shape)

    train_mnist(config,(data_x,data_y))
 


if(__name__=="__main__"):

    model_name = sys.argv[1]
    if(model_name== 'xgb'):
        train_func = xgb_explore
    elif(model_name=='keras'):
        train_func = tf_explore
    elif(model_name=='torch'):
        train_func = torch_explore
    else:
        print("This Model does not exist, select [xgb,torch,keras]")
        raise NotImplementedError

    train_data, _, _ = read_mnist(False,False,0.0) #not using test data for hp search
    params_filename = f'{model_name}_params.pkl' #output filename with params

    train_x,train_y = train_data
    train_x,train_y = train_x[:50], train_y[:50] 

    num_classes  = len(set(train_y))
    input_size = train_x.shape

    #backing up old parameter search (if there is one)
    previous_exists = os.path.isfile(params_filename)
    if(previous_exists):
        print(f"[Warning] Renaming {params_filename} to {params_filename}.bkp")
        os.rename(params_filename, params_filename+".bkp")
    
    #hyperparam search
    best_params = train_func(train_x,train_y) #finding best params and saving

    print(best_params)
    #TODO save best params with pickle