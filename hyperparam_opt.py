import os, sys, json
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


num_trials = 20#20
EPOCHS = 2 #10
BATCH_SIZE = 128

#k-fold search using sklearn
def xgb_explore(data_x,data_y):

    xgb_num_threads = 4
    xgb_model = xgb.XGBClassifier(seed=20)

    #exploration space
    params ={ 'max_depth': [3,6,10],
        'min_child_weight': [1, 5, 10],
        'gamma': [0,0.5, 1, 5],
        'learning_rate': [0.001, 0.01,0.1,0.2,0.3],
        'n_estimators': [100, 500, 1000],
        'subsample': [0.3, 0.5, 0.8],
        'colsample_bytree': [0.3, 0.5, 0.8],
        'nthread':[xgb_num_threads]} #num of threads in a single xgboost execution
    
    #.Random Search
    tune = sklearn.model_selection.RandomizedSearchCV(xgb_model, 
            params, 
            n_iter=num_trials, 
            scoring='accuracy', #balanced_accuracy_score 
            n_jobs=int(cpu_count()/xgb_num_threads), #num of parallel trials (multi-xgb)
            verbose=3
            )

    #handling the data
    data_x = flatten(data_x)
    tune.fit(data_x,data_y)
    print("Best parameters:", tune.best_params_)
    best_params = tune.best_params_.copy()


    return best_params

#TODO K-fold
#random search using keras_tuner
def tf_explore(data_x,data_y):

    def cnn_builder(params):

        model = tf.keras.models.Sequential()

        hp_cnn_input = params.Int('cnn_0', min_value=32, max_value=256, step=32, default=32) #num filters
        hp_dropout_0 = params.Float('dropout_0', min_value=0.1, max_value=0.5, step=0.1, default=0.5) #dropout
        hp_cnn_num = params.Int('num_conv',min_value=0,max_value=3,step=1,default=0) #number of conv layers
        hp_dense_num = params.Int('num_dense',min_value=0,max_value=3,step=1,default=1) #number of dense layers

        model.add(tf.keras.layers.Conv2D(hp_cnn_input, input_shape=(data_x.shape[1:]), kernel_size=(3, 3), activation="relu")) #could additionally search kernel_size, stride and other params
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))) #could also vary pooling types and sizes

        #searching subsequent cnn layers
        for i in range(hp_cnn_num):

            hp_cnn = params.Int(f'cnn_{i+1}', min_value=32, max_value=256, step=32, default=128)

            model.add(tf.keras.layers.Conv2D(hp_cnn, kernel_size=(3, 3), activation="relu")) #could additionally search kernel_size, stride and other params
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        #flattening output of CNN layers
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(hp_dropout_0))

        #searching FC layers
        for i in range(hp_dense_num): 

            hp_dense_units = params.Int(f'dense_{i+1}', min_value=16, max_value=128, step=16, default=64)
            hp_dropout = params.Float(f'dropout_{i+1}', min_value=0.1, max_value=0.5, step=0.1, default=0.2)

            model.add(tf.keras.layers.Dense(hp_dense_units, activation='relu'))
            model.add(tf.keras.layers.Dropout(hp_dropout))

        model.add(tf.keras.layers.Dense(10, activation="softmax")) #output layer

        hp_learning_rate = params.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])
        #hp_learning_rate = hp.Float('learning_rate', 1e-4, 1e-1, sampling='log')
        opt = tf.keras.optimizers.Adam(lr=hp_learning_rate, decay=1e-6)

        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        return model


    #handling the data for this model
    data_x = data_x.astype("float32") / 255
    data_x = np.expand_dims(data_x, -1)
    data_y = tf.keras.utils.to_categorical(data_y, 10)

    tuner = kt.RandomSearch(cnn_builder,
        objective='val_accuracy',
        max_trials=num_trials, #search more combinations
        directory='tune_param',
        project_name=f'MNIST_keras'
        )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, mode='max', restore_best_weights=True)
    tuner.search(data_x, data_y, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[stop_early])
    tuner.results_summary(num_trials=10) #print top 10 trials
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    
    best_params = {'learning_rate':best_hps.get('learning_rate'),'num_conv':best_hps.get('num_conv'), 'num_dense':best_hps.get('num_dense')}
    best_params[f'dropout_0'] = best_hps.get(f'dropout_0')
    for i in range(0,best_hps.get('num_conv')+1):
        best_params[f'cnn_{i}'] = best_hps.get(f'cnn_{i}')
    for i in range(1,best_hps.get('num_dense')+1):
        best_params[f'dense_{i}'] = best_hps.get(f'dense_{i}')
        best_params[f'dropout_{i}'] = best_hps.get(f'dropout_{i}')

    #hypermodel = tuner.hypermodel.build(best_hps) #instantiate model with best parameters

    return best_params

#TODO K-fold
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
            tune.report(loss=loss.item(), accuracy=val_acc) #!reporting back to raytune
            
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
        
    dense = [j for j in range(16,128,16)]
    cnn = [j for j in range(32,256,32)]
    dropout = [j/10 for j in range(1,5,1)]

    #filling radom search
    config = {
            "cnn_0": tune.choice(cnn),
            "dropout_0": tune.choice(dropout),
            "num_conv": tune.choice([i for i in range(0,3,1)]),
            "num_dense": tune.choice([i for i in range(0,3,1)]),

            "lr": tune.loguniform(1e-4, 1e-1),
            #"batch_size": tune.choice([16,32,64,128]),
            #"epochs": tune.grid_search([5,10,15]),
            
        }
    
    for i in range(1,4):
        config[f'dense_{i}'] = tune.choice(dense)
        config[f'dropout_{i}'] = tune.choice(dropout)
        config[f'cnn_{i}'] = tune.choice(cnn)

    scheduler = tune.schedulers.ASHAScheduler(
        metric='accuracy',
        mode="max",
        #max_t=100,
        #grace_period=10,
        reduction_factor=2)

    reporter = tune.CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
    
    data_x = data_x.astype("float32") / 255
    data_x = np.expand_dims(data_x, 1)

    result = tune.run(
        tune.with_parameters(train_mnist, data=(data_x,data_y)),
        resources_per_trial={"cpu": cpu_count()}, #using cpu only
        config=config,
        num_samples=num_trials,
        scheduler=scheduler,
        progress_reporter=reporter)
        #log_to_file=(f"logs/tune_stdout.log", f"logs/tune_stderr.log")

    best_trial = result.get_best_trial("loss", "min", "last")
    best_params = dict(best_trial.config)

    return best_params

 


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
    params_filename = f'params/{model_name}_params.json' #output filename with params

    train_x,train_y = train_data

    num_classes  = len(set(train_y))
    input_size = train_x.shape

    #backing up old parameter search (if there is one)
    previous_exists = os.path.isfile(params_filename)
    if(previous_exists):
        print(f"[Warning] Renaming {params_filename} to {params_filename}.bkp")
        os.rename(params_filename, params_filename+".bkp")
    
    #hyperparam search
    best_params = train_func(train_x,train_y) #finding best params and saving

    #saving best params
    print(f"Saving best params for {model_name} . . .")
    print(best_params)
    with open(params_filename, 'w') as f:
        json.dump(best_params, f)
