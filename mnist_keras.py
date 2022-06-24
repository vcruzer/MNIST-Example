from tensorflow import keras
import utils_dataset as ud
from sklearn.metrics import accuracy_score
import numpy as np


EPOCHS = 10
BATCH_SIZE = 128

def model_arch(input_shape,num_classes):

    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32, input_shape=(input_shape[1:]), kernel_size=(3, 3), activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    return model

if(__name__=="__main__"):

    #loading mnist dataset
    train_data, val_data, test_data = ud.read_mnist()

    train_x,train_y = train_data
    val_x, val_y = val_data
    test_x, test_y = test_data

    num_classes  = len(set(train_y))
    input_size = train_x.shape

    train_y = keras.utils.to_categorical(train_y, num_classes)
    val_y = keras.utils.to_categorical(val_y, num_classes)
    #test_y = keras.utils.to_categorical(test_y, num_classes)
 
    #defining the model
    model = model_arch(input_size,num_classes)

    opt = keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    #training the model
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, mode='max', restore_best_weights=True)
    history = model.fit(
        train_x, train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_x,val_y),
        verbose=1,
        #callbacks = [tboard_callback]
        callbacks=[early_stop]
    )

    #score = model.evaluate(test_x,test_y, verbose=1)
    ypred = model.predict(test_x)
    ypred = np.argmax(ypred,axis=1)

    print(f"Test Score: {accuracy_score(test_y, ypred)}")
