import numpy as np 
from matplotlib import pyplot as plt
import tensorflow as tf 
from tensorflow.keras import layers, utils
from tqdm import tqdm
import argparse

def siren_model(n_neurons=128,n_layers=3):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(n_neurons, input_shape=(6,), activation=tf.math.sin))
    for i in range(n_layers):
        model.add(tf.keras.layers.Dense(n_neurons, activation=tf.math.sin))
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    model.compile(loss='mse', optimizer='Adam')
    return model

def log_1px_transform(x):
    return np.log(1+x)

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')

    argparser.add_argument('--lr', type=float, default=1e-3, help='training rate')
    argparser.add_argument('--n_layers',  type=int, default=4,   help='number of siren layers')
    argparser.add_argument('--n_neurons',  type=int, default=64,   help='number of neurons')   
    argparser.add_argument('--batch_size',  type=int, default=1024,   help='batch size')
    argparser.add_argument('--epochs',  type=int, default=25,   help='number of epochs')   
    argparser.add_argument('--data_path', type=str, default='src/paper_data_code/neural_quantum_data2.npz', help='data directory for training')
    argparser.add_argument('--model_path', type=str, default='src/paper_data_code/siren_model', help='filename for trained model')
        
    args = argparser.parse_args()
    data = np.load(args.data_path)

    X_train = data['train_x']
    X_valid = data['valid_x']
    X_test = data['test_x']
    
    y_train = log_1px_transform(data['train_y'])
    y_valid = log_1px_transform(data['valid_y'])
    y_test = log_1px_transform(data['test_y'])
    
    print(" -------------------------------- Dataset loaded -------------------------------- ")
    
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( filepath=args.model_path, 
                                                                    save_weights_only=False,
                                                                    monitor='val_loss',
                                                                    mode='min',
                                                                    save_best_only=True)
    
    
    
    model = siren_model(n_neurons=args.n_neurons, n_layers=args.n_layers)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer, loss=loss)
    print(model.summary())
    
    print(" -------------------------------- Begin model training -------------------------------- ")
        
    history = model.fit(X_train,
                        y_train, 
                        validation_data = (X_valid, y_valid), 
                        verbose = 1, 
                        batch_size=args.batch_size, 
                        epochs=args.epochs, 
                        callbacks=[lr_schedule, model_checkpoint_callback])
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()
