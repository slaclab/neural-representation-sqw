import numpy as np
import tensorflow as tf 
from tensorflow.keras import layers, utils
import tensorflow_probability as tfp
import cv2 
from scipy.signal import savgol_filter
from tqdm import tqdm 
from matplotlib import pyplot as plt 

def path2mesh_expt(j1, j2, c_q, c_E, model):
                   
    mesh = [] 
    
    for j in range(c_E.shape[0]):
        for i in range(c_q.shape[0]):
            mesh.append([c_q[i][0], c_q[i][1], c_q[i][2], float(c_E[j])/200])
            
    x = np.array(mesh)
    j1v = np.expand_dims(j1 * np.ones(len(x)),axis=-1)
    j2v = np.expand_dims(j2 * np.ones(len(x)),axis=-1)
    
    x = np.hstack((x,j1v))
    x = np.hstack((x,j2v))

    y_pred = model.predict(x)
    y_img = np.reshape(y_pred, (c_E.shape[0], c_q.shape[0]))

    return y_img 

def generate_background(c_sqw, start, end):
    bkg = np.mean(c_sqw.T[:,start:end],axis=1)
    yhat = savgol_filter(bkg, 51, 3) # window size 51, polynomial order 3
    BKG = (np.expand_dims(yhat,axis=-1) * np.expand_dims(np.ones(c_sqw.shape[0]),axis=0)).T
    return BKG

def correlation_loss(y_true, y_pred):
    y_true_flat = layers.Flatten(name="Y_TRUE_FLAT")(y_true)
    y_pred_flat = layers.Flatten(name="Y_PRED_FLAT")(y_pred)
    cov =  tfp.stats.covariance(y_true_flat, y_pred_flat, sample_axis=0, event_axis=None, name="COVARIANCE")
    std_y_trueR = tfp.stats.stddev(y_true_flat, sample_axis=0, keepdims=False, name="LOSS_STD_TRUE")
    std_y_predR = tfp.stats.stddev(y_pred_flat, sample_axis=0, keepdims=False, name="LOSS_STD_PRED")
    corr = tf.math.divide(cov,tf.math.multiply(std_y_trueR,std_y_predR, name="MULT_STDs"), name="CORRELATION")
    loss = tf.math.subtract(1.0,corr[0], name="CORR_LOSS")
    return loss

def rejection_sampling(c_sqw, n_samples = 25000, filter_size=3):
    
    np.random.seed(47)
    
    Gaussian = cv2.GaussianBlur(c_sqw.T, (filter_size, filter_size), 0)    
    Z = np.sum(Gaussian)
    p_true = Gaussian/Z
    
    uniform_discrete = np.ones(c_sqw.T.shape)
    uniform_discrete = uniform_discrete/np.sum(uniform_discrete)
    
    C = np.max(p_true)/np.max(uniform_discrete)
    sampled = np.zeros(c_sqw.T.shape)
    for i in range(n_samples):
        idx = np.random.randint(p_true.shape)
        u = np.random.rand()
        fx = p_true[idx[0]][idx[1]]
        gx = uniform_discrete[idx[0]][idx[1]]

        if u  < fx/(C * gx):
            sampled[idx[0]][idx[1]] += 1
            
    return sampled

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
    
    
def image_to_coords(c_q, c_E, c_sqw, background_start=150,background_end=160):
    
    test_x = []
    test_y = [] 

    c_sqw_subtract_bkg = c_sqw - generate_background(c_sqw, background_start, background_end)
    c_sqw_subtract_bkg[c_sqw_subtract_bkg < 0] = 0
    
    for j in range(c_sqw_subtract_bkg.shape[1]):
        for i in range(c_sqw_subtract_bkg.shape[0]):
            test_x.append([c_q[i][0], c_q[i][1], c_q[i][2], float(c_E[j])/200])
            test_y.append(c_sqw_subtract_bkg[i,j])
        
    test_x = np.array(test_x) 
    test_y = np.array(test_y)

    return test_x, test_y

def calculate_loss_landscape(test_x, test_y, gridsize = 50):
    
    ones_vector = tf.ones(test_x.shape[0])

    loss_vals = []

    # Define a uniform grid to evaluate loss function
    j1 = np.linspace(-0.5, 0.5, gridsize)
    j2 = np.linspace(0.0, 1.5, gridsize)
    
    j1j1, j2j2 = np.meshgrid(j1,j2)
    j1_all = np.ravel(j1j1)
    j2_all = np.ravel(j2j2)

    for i in tqdm(range(len(j1_all))):
        j1 = j1_all[i]
        j2 = j2_all[i]

        j1_vector = tf.expand_dims(j1 * ones_vector,axis=-1)
        j2_vector = tf.expand_dims(j2 * ones_vector,axis=-1)

        x_in = tf.concat((test_x, j1_vector, j2_vector),axis=1)
        y_pred = model.predict(x_in)

        loss = correlation_loss(test_y, y_pred)
        loss_vals.append([j1_all[i], j2_all[i], float(loss)])

    loss_vals = np.array(loss_vals)
    
    return loss_vals

def optimize_surrogate(test_x, test_y, model, learning_rate = 0.01, batch_size = 2048, max_iter=2000, plotting = True):
    loss_vals = [] 
    metrics = [] 
    
    # Adam optimizer 
    opt = tf.optimizers.Adam(learning_rate)

    j1 = tf.Variable(tf.random.uniform(shape=[1],minval=-0.5,maxval=0.5), constraint=lambda t: tf.clip_by_value(t, -0.5, 0.5))
    j2 = tf.Variable(tf.random.uniform(shape=[1],minval=0.0,maxval=1.5), constraint=lambda t: tf.clip_by_value(t, 0.0, 1.5))
    
    for i in range(max_iter):

        ones_vector = tf.ones(test_x.shape[0])

        with tf.GradientTape() as tape:

            tape.watch(j1)
            tape.watch(j2)

            j1_vector = tf.expand_dims(j1 * ones_vector,axis=-1)
            j2_vector = tf.expand_dims(j2 * ones_vector,axis=-1)

            # forward model 
            x_in = tf.concat((test_x, j1_vector, j2_vector),axis=1)
            ridxs = tf.convert_to_tensor(np.random.choice(np.arange(0, len(x_in)), batch_size))
            x_in_batch = tf.gather(x_in, ridxs)

            y_pred_batch =  tf.clip_by_value(model(x_in_batch), 0, 10000)
            test_y_batch = tf.expand_dims(tf.math.log(tf.clip_by_value(tf.gather(test_y, ridxs),0,10000) + 1),axis=-1)

            loss = correlation_loss(y_pred_batch, test_y_batch)

        gradients = tape.gradient(loss, [j1, j2])
        opt.apply_gradients(zip(gradients, [j1, j2]))
        
        
        metrics.append([float(loss), float(j1), float(j2)])

    metrics = np.array(metrics)

    if plotting:
        plt.plot(np.arange(0,max_iter), 1 - metrics[:,0], alpha=0.9)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Correlation function")
        plt.show()

        plt.plot(np.arange(0,max_iter), metrics[:,1], alpha=0.9)
        plt.xlabel("Number of Iterations")
        plt.ylabel("J1")
        plt.show()

        plt.plot(np.arange(0,max_iter), metrics[:,2], alpha=0.9)
        plt.xlabel("Number of Iterations")
        plt.ylabel("J2")
        plt.show()
    
    min_loss, min_loss_j1, min_loss_j2 = metrics[np.argmin(metrics[:,0])]

    return min_loss, min_loss_j1, min_loss_j2, metrics 