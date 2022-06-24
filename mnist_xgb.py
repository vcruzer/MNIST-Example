import xgboost as xgb
from utils_dataset import read_mnist
from multiprocessing import cpu_count
from sklearn.metrics import accuracy_score

NUM_ROUNDS = 15

def define_params(num_classes):
    param = {}

    param['booster'] = 'gbtree' #gbtree, dart, gblinear # gbtree and dart use tree based models while gblinear uses linear functions.
    param['nthread'] = cpu_count()
    param['eval_metric'] = ['merror']
    param['objective'] = 'multi:softmax'
    param['num_class'] = num_classes

    return param 

def flatten(arr):

    shape = arr.shape
    
    return arr.reshape((shape[0],shape[1]*shape[2]))
    

if(__name__=="__main__"):

    train_data, val_data, test_data = read_mnist(ch_dim=False,norm=False)

    train_x,train_y = train_data
    val_x, val_y = val_data
    test_x, test_y = test_data

    num_classes = len(set(train_y))

    #flattening
    train_x = flatten(train_x)
    val_x = flatten(val_x)
    test_x = flatten(test_x)

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dval = xgb.DMatrix(val_x, label=val_y)
    dtest = xgb.DMatrix(test_x)

    params = define_params(num_classes)
    evallist = [(dval, 'eval'), (dtrain, 'train')]
    progress = dict()

    model = xgb.train(params, dtrain, NUM_ROUNDS, evals=evallist, evals_result=progress, early_stopping_rounds=5)

    ypred = model.predict(dtest)

    print(f"Test Score: {accuracy_score(test_y, ypred)}")