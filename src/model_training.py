import numpy as np 
from matplotlib import pyplot as plt
import tensorflow as tf 
from tqdm import tqdm
import argparse
from utils import *

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser(description='Train ML model with specific hparams')

    argparser.add_argument('--lr', type=float, default=1e-3, help='training rate')
    argparser.add_argument('--n_layers',  type=int, default=4,   help='number of siren layers')
    argparser.add_argument('--n_neurons',  type=int, default=64,   help='number of neurons')   
    argparser.add_argument('--batch_size',  type=int, default=1024,   help='batch size')
    argparser.add_argument('--epochs',  type=int, default=30,   help='number of epochs')   
    argparser.add_argument('--data_path', type=str, default='src/training_data/siren_training_data.npz', help='data directory for training')
    argparser.add_argument('--model_path', type=str, default='surrogate_model', help='filename for trained model')
        
    args = argparser.parse_args()
    
    # Load training dataset
    data = np.load(args.data_path)

    X_train = data['train_x']
    X_valid = data['valid_x']
    X_test = data['test_x']
    
    y_train = log_1px_transform(data['train_y'])
    y_valid = log_1px_transform(data['valid_y'])
    y_test = log_1px_transform(data['test_y'])
    
    print(" -------------------------------- Dataset loaded -------------------------------- ")
    
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.model_path, save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss = tf.keras.losses.MeanSquaredError()
    
    model = siren_model(n_neurons=args.n_neurons, n_layers=args.n_layers)
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
    
    np.save(args.model_path + "_training_loss", history.history['loss'])
    np.save(args.model_path + "_validation_loss", history.history['val_loss'])
    
    model.save(args.model_path)

    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()
