from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split

#downloads and reads mnist dataset using keras
#returns: list of train, validation and test data
def read_mnist(ch_dim=True, norm=True,split_pct=0.2):
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()

    if(norm):
        train_x = train_x.astype("float32") / 255
        test_x = test_x.astype("float32") / 255

    #adds channel dim at the end
    if(ch_dim):
        train_x = np.expand_dims(train_x, -1)
        test_x = np.expand_dims(test_x, -1)

    if(split_pct > 0.0):
        train_x, val_x, train_y, val_y = train_test_split(train_x,train_y, test_size=split_pct, shuffle=True)
    else:
        val_x, val_y = None, None

    train_data = [train_x,train_y]
    val_data = [val_x,val_y]
    test_data = [test_x,test_y]
    
    return train_data, val_data, test_data

