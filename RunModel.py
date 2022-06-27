import sys
import os
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
import scipy.io as spio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pycm import *
# import the garbage colector to clear the memory
import gc
# importing the library to save the results
import pickle
# importing the library to check the dataset files
from os import path
import pickle 
import csv
import matplotlib.pyplot as plt
import os
import warnings
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import pickle
from scipy.io import savemat
from pycm import *
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from sklearn.metrics import mean_squared_error
tfd = tfp.distributions
tfpl = tfp.layers


#--------------------- 1 dimension LeNet-AlexNet- Train(edit) -----------------

#-------------------------Run 1 dimension LeNet  ------------------------------

def runmodel_lenet_1d(SAMPLING_FREQ=200, EXAMINE_AVERAGE = 20, THRESHOLD_EARLY_STOPING = 0.005,
                   BATCH_SIZE=32, NUMBER_EPOCHS = 400, PATIENCE_VALUE=40, classes=5):
    print("LeNet Model is running!")
    print("\nThe parameters that you choose are:")
    print("\Input frequency:", SAMPLING_FREQ) 
    print("Monte Carlo:", EXAMINE_AVERAGE) 
    print("Patience value", THRESHOLD_EARLY_STOPING) 
    print("Batch size:", BATCH_SIZE)  
    print("Maximum number of epoch:", NUMBER_EPOCHS) 
    print("Patience value", PATIENCE_VALUE) 
    print("\nLoading x_train_data.csv - Can take serveral minutes")
    #Load the x_train_data
    x_train_data = np.genfromtxt("x_train_data.csv")
    print("x_train_data.csv loaded sucessfuly")
    print("\nLoading x_test_data.csv - Can take serveral minutes")
    x_test_data = np.genfromtxt("x_test_data.csv")
    print("x_test_data.csv loaded sucessfuly")
    print("\nLoading y_train_label.csv - Can take serveral minutes")
    y_train_label = np.genfromtxt("y_train_label.csv")
    print("y_train_label.csv loaded sucessfuly")
    print("\nLoading y_test_label.csv - Can take serveral minutes")
    y_test_label = np.genfromtxt("y_test_label.csv")
    print("y_test_label.csv loaded sucessfuly")    
    
 #Define the helping function for the training
    def train_model (x_train, y_train, x_valid, y_valid, model, class_weights):
        # clear the memory
        gc.collect()
        # class to perform the early stopping
        class EarlyStoppingAtMinLoss (tf.keras.callbacks.Callback): 
            # initialization of the class
            def __init__ (self, PATIENCE_VALUE, valid_data): 
                # patience value for the early stopping procedure, defining the maximum number of iteration without an increasse of at least "THRESHOLD_EARLY_STOPING" in the Acc before stopping the training procedure
                super (EarlyStoppingAtMinLoss, self).__init__ ()
                self.patience = PATIENCE_VALUE 
                # best weights of the network
                self.best_weights = None 
                # data used to validate the model
                self.validation_data = valid_data  
    
            # initialize the control parametrers
            def on_train_begin (self, logs = None): 
                # variable holding the number of training iterations without an increasse
                self.wait = 0 
                # variable hold the value of the training epoch where the model early stoped
                self.stopped_epoch = 0 
                # initialization of the variable holding the identified best Acc
                self.best = 0.
                # variable holding the data
                self._data = [] 
                # initialization of the variable holding the Acc of the curent training epoch
                self.curent_Acc = 0.   
    
            # examination at the end of a training epoch
            def on_epoch_end (self, epoch, logs = None): 
                # load the validation data 
                x_val, y_val = self.validation_data [0], self.validation_data [1]   
                # define variable to hold the results with shape: number of examples, number of classes, number of examinations
                probabilistic_output_x_val = np.zeros([len(x_val), 5, EXAMINE_AVERAGE])
                # calculate multiple times the results to assess the epistemic uncertainty 
                for testings in range (0,EXAMINE_AVERAGE,1):
                    # clear GPU memory
                    tf.keras.backend.clear_session()
                    gc.collect()
                    probabilistic_output_x_val[:,:,testings] = model.predict(x_val)
                # estimate the average to determine to remove the epistemic variations from the predictions
                # convert to a single column array with argmax and estimate the accuracy             
                self.curent_Acc = accuracy_score(np.argmax (y_val, axis = 1), np.argmax (np.mean (probabilistic_output_x_val, axis = 2), axis = 1))
                # save the weights if the current Acc is at lest "THRESHOLD_EARLY_STOPING" better than the preivously identifed best Acc
                if np.greater(self.curent_Acc, self.best + THRESHOLD_EARLY_STOPING): 
                    # update the currently best Acc
                    self.best = self.curent_Acc 
                    # restart the counting variable for the early stopping procedure
                    self.wait = 0 
                    # save the weights of the identified best model
                    self.best_weights = self.model.get_weights () 
                else: 
                    # increasse the counting variable for the early stopping procedure
                    self.wait += 1 
                    # early stop the training if the number of training epochs without a minimum Acc increasse of "THRESHOLD_EARLY_STOPING" was higher than the defined patience value
                    if self.wait >= self.patience: 
                        # save the training epoch were the model early stopped
                        self.stopped_epoch = epoch 
                        # flag to identify an early stop
                        self.model.stop_training = True 
                        # restore the weights of the identified best
                        self.model.set_weights (self.best_weights)  
    
            # precedure performed at the end of the training
            def on_train_end (self, logs = None): 
                # report if early stopping occured
                if self.stopped_epoch > 0: 
                    print ('Epoch %05d: early stopping' % (self.stopped_epoch + 1))               
        
        # fit the model using the train dataset, with batch size and number of epochs defined by the user,
        # validate the model at the end of each training epoch with the validation dataset
        # print the results of each training epoch with the verbose set to 1
        # pass the class_weights to use const sensitive learning
        # shuffle the dataset at the end of each training epoch
        # set the callbacks to perform the early stopping
        history = model.fit (x_train, y_train,
                          batch_size = BATCH_SIZE,
                          epochs = NUMBER_EPOCHS,
                          validation_data = (x_valid, y_valid),
                          verbose = 1,
                          class_weight = class_weights,
                          shuffle = True, 
                          callbacks = EarlyStoppingAtMinLoss (PATIENCE_VALUE, (x_valid, y_valid)))
        
        
        # return the trained model
        return (model, history)
    # Define the model
    # create the KL divergence function
    divergence_fn = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                              tf.cast(x_train_data.shape[0], dtype=tf.float32))
    
    # define the model using the sequential API
    model = Sequential()
    model.add(tfpl.Convolution1DFlipout(
        filters = 6,
        kernel_size = 5,
        strides = 1,
        activation = 'relu',
        input_shape = (SAMPLING_FREQ * 30, 1),
        kernel_prior_fn = tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        kernel_divergence_fn = divergence_fn, 
        bias_prior_fn = tfpl.default_multivariate_normal_fn,
        bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        bias_divergence_fn = divergence_fn
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Dropout(0.2))
    model.add(tfpl.Convolution1DFlipout(
        filters = 16,
        kernel_size = 5,
        strides = 1,
        activation = 'relu',
        kernel_prior_fn = tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        kernel_divergence_fn = divergence_fn, 
        bias_prior_fn = tfpl.default_multivariate_normal_fn,
        bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        bias_divergence_fn = divergence_fn
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(tfpl.DenseFlipout(
        units = 120, 
        activation = 'relu',
        kernel_prior_fn = tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        kernel_divergence_fn = divergence_fn, 
        bias_prior_fn = tfpl.default_multivariate_normal_fn,
        bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        bias_divergence_fn = divergence_fn
    ))
    model.add(tfpl.DenseFlipout(
        units = 84, 
        activation = 'relu',
        kernel_prior_fn = tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        kernel_divergence_fn = divergence_fn, 
        bias_prior_fn = tfpl.default_multivariate_normal_fn,
        bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        bias_divergence_fn = divergence_fn
    ))
    model.add(tfpl.DenseFlipout(
        units = classes, 
        activation = 'softmax',
        kernel_prior_fn = tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        kernel_divergence_fn = divergence_fn, 
        bias_prior_fn = tfpl.default_multivariate_normal_fn,
        bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        bias_divergence_fn = divergence_fn
    ))    
    
    #%% compile, train and test the model
    
    # recommend to change for two fold cross-validation
            
    # compile the model with the ADAM algorithm
    # Keras API automatically adds the Kullback-Leibler divergence (contained on the individual layers of the model) to the cross entropy loss, effectivly computing the ELBO loss
    # track the model accuracy and run witouth the experimental run
    model.compile (optimizer = 'adam', loss = 'categorical_crossentropy',
                  metrics = ['accuracy'], experimental_run_tf_function = False)            
    
    # compute the class wights to be balanced
    class_weights = class_weight.compute_class_weight('balanced',
                                              np.unique(y_train_label),
                                              y_train_label)
    class_weights = {i : class_weights[i] for i in range(classes)}
    
    # the labels are for each 30 s of data so reshape the dataset to have 30 s of data for each label
    # every second of data has 200 points (sampling freqency of 200 Hz) * 30 s of data
    # reshape to the standard: number of examples, number of time steps, number of features for each time step
    x_train_data = x_train_data.reshape (round (len(x_train_data) / (SAMPLING_FREQ * 30)), SAMPLING_FREQ * 30, 1) 
    
    # produce the validation dataset 
    x_train, x_valid, y_train, y_valid = train_test_split (x_train_data, y_train_label, test_size = 0.33, shuffle = True)
    
    # convert the labels to categorical
    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)
    
    # train the model
    model, history = train_model (x_train, y_train, x_valid, y_valid, model, class_weights)
    
    # Monte Carslo sample the model using the test dataset
    for _ in range(EXAMINE_AVERAGE):
        
        # test the model and return the predictions
        model_prodictions = model.predict (x_test_data.reshape (round (len(x_test_data) / (SAMPLING_FREQ * 30)), SAMPLING_FREQ * 30, 1))
        # convert to a single column array with argmax and estimate the performance metrics    
        m = ConfusionMatrix (actual_vector = np.asarray (y_test_label, dtype = int), predict_vector = np.argmax (model_prodictions, axis=1))

        Acc_class = m.ACC
        Sen_class = m.TPR
        Spe_class = m.TNR
        AUC_class = m.AUC
        PPV_class = m.PPV
        NPV_class = m.NPV
        # save the results in a pickle
        f = open("Results\LeNet1d\m1.txt", 'ab')
        pickle.dump(m, f)
        f.close()
        f = open("Results\LeNet1d\AccLenet1.txt", 'ab')
        pickle.dump(Acc_class, f)
        f.close()
        f = open("Results\LeNet1d\SenLenet1.txt", 'ab')
        pickle.dump(Sen_class, f)
        f.close()
        f = open("Results\LeNet1d\SpeLenet1.txt", 'ab')
        pickle.dump(Spe_class, f)
        f.close()
        f = open("Results\LeNet1d\AucLenet1.txt", 'ab')
        pickle.dump(AUC_class, f)
        f.close()
        f = open(rb"Results\LeNet1d\PpvLenet1.txt", 'ab')
        pickle.dump(PPV_class, f)
        f.close()
        f = open(rb"Results\LeNet1d\NpvLenet1.txt", 'ab')
        pickle.dump(NPV_class, f)
        f.close()
        f = open(rb"Results\LeNet1d\all_metrics.txt", 'ab')
        pickle.dump(m, f)
        f.close()  
        #model.save("LeNet1d\model_LeNet1d")

    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid() 
    plt.show()
    plt.savefig("Results\LeNet1d\Accuracy.png")           
    
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'validation'], loc='upper left')
    plt.grid() 
    plt.show()
    plt.savefig("Results\LeNet1d\Loss.png")    

    from pretty_confusion_matrix import pp_matrix_from_data
    cmap = 'copper'
    pp_matrix_from_data(np.asarray (y_test_label, dtype = int), np.argmax (model_prodictions, axis=1) , cmap=cmap)
    plt.savefig("Results\LeNet1d\ConfusionMatrix.png")
    #model.save("Results\LeNet1d\model_LeNet1d")  
    print(model.summary())
    
#-------------------------Run 1 dimension  AlexNet  ------------------------------

def runmodel_alexnet(SAMPLING_FREQ=200, EXAMINE_AVERAGE = 20, THRESHOLD_EARLY_STOPING = 0.005,
                   BATCH_SIZE=32, NUMBER_EPOCHS = 400, PATIENCE_VALUE=40, classes=5):
    
    print("Alexnet Model is running!")
    print("\nThe parameters that you choose are:")
    print("\Input frequency:", SAMPLING_FREQ) 
    print("Monte Carlo:", EXAMINE_AVERAGE) 
    print("Patience value", THRESHOLD_EARLY_STOPING) 
    print("Batch size:", BATCH_SIZE)  
    print("Maximum number of epoch:", NUMBER_EPOCHS) 
    print("Patience value", PATIENCE_VALUE) 
    print("\nLoading x_train_data.csv - Can take serveral minutes")
    x_train_data = np.genfromtxt("x_train_data.csv")
    print("x_train_data.csv loaded sucessfuly")
    print("\nLoading x_test_data.csv - Can take serveral minutes")
    x_test_data = np.genfromtxt("x_test_data.csv")
    print("x_test_data.csv loaded sucessfuly")
    print("\nLoading y_train_label.csv - Can take serveral minutes")
    y_train_label = np.genfromtxt("y_train_label.csv")
    print("y_train_label.csv loaded sucessfuly")
    print("\nLoading y_test_label.csv - Can take serveral minutes")
    y_test_label = np.genfromtxt("y_test_label.csv")
    print("y_test_label.csv loaded sucessfuly")    
    #Define the helping function for the training
    def train_model (x_train, y_train, x_valid, y_valid, model, class_weights):
        # clear the memory
        gc.collect()
        # class to perform the early stopping
        class EarlyStoppingAtMinLoss (tf.keras.callbacks.Callback): 
            # initialization of the class
            def __init__ (self, PATIENCE_VALUE, valid_data): 
                # patience value for the early stopping procedure, defining the maximum number of iteration without an increasse of at least "THRESHOLD_EARLY_STOPING" in the Acc before stopping the training procedure
                super (EarlyStoppingAtMinLoss, self).__init__ ()
                self.patience = PATIENCE_VALUE 
                # best weights of the network
                self.best_weights = None 
                # data used to validate the model
                self.validation_data = valid_data  
    
            # initialize the control parametrers
            def on_train_begin (self, logs = None): 
                # variable holding the number of training iterations without an increasse
                self.wait = 0 
                # variable hold the value of the training epoch where the model early stoped
                self.stopped_epoch = 0 
                # initialization of the variable holding the identified best Acc
                self.best = 0.
                # variable holding the data
                self._data = [] 
                # initialization of the variable holding the Acc of the curent training epoch
                self.curent_Acc = 0.   
    
            # examination at the end of a training epoch
            def on_epoch_end (self, epoch, logs = None): 
                # load the validation data 
                x_val, y_val = self.validation_data [0], self.validation_data [1]   
                # define variable to hold the results with shape: number of examples, number of classes, number of examinations
                probabilistic_output_x_val = np.zeros([len(x_val), 5, EXAMINE_AVERAGE])
                # calculate multiple times the results to assess the epistemic uncertainty 
                for testings in range (0,EXAMINE_AVERAGE,1):
                    # clear GPU memory
                    tf.keras.backend.clear_session()
                    gc.collect()
                    probabilistic_output_x_val[:,:,testings] = model.predict(x_val)
                # estimate the average to determine to remove the epistemic variations from the predictions
                # convert to a single column array with argmax and estimate the accuracy             
                self.curent_Acc = accuracy_score(np.argmax (y_val, axis = 1), np.argmax (np.mean (probabilistic_output_x_val, axis = 2), axis = 1))
                # save the weights if the current Acc is at lest "THRESHOLD_EARLY_STOPING" better than the preivously identifed best Acc
                if np.greater(self.curent_Acc, self.best + THRESHOLD_EARLY_STOPING): 
                    # update the currently best Acc
                    self.best = self.curent_Acc 
                    # restart the counting variable for the early stopping procedure
                    self.wait = 0 
                    # save the weights of the identified best model
                    self.best_weights = self.model.get_weights () 
                else: 
                    # increasse the counting variable for the early stopping procedure
                    self.wait += 1 
                    # early stop the training if the number of training epochs without a minimum Acc increasse of "THRESHOLD_EARLY_STOPING" was higher than the defined patience value
                    if self.wait >= self.patience: 
                        # save the training epoch were the model early stopped
                        self.stopped_epoch = epoch 
                        # flag to identify an early stop
                        self.model.stop_training = True 
                        # restore the weights of the identified best
                        self.model.set_weights (self.best_weights)  
    
            # precedure performed at the end of the training
            def on_train_end (self, logs = None): 
                # report if early stopping occured
                if self.stopped_epoch > 0: 
                    print ('Epoch %05d: early stopping' % (self.stopped_epoch + 1))               
        
        # fit the model using the train dataset, with batch size and number of epochs defined by the user,
        # validate the model at the end of each training epoch with the validation dataset
        # print the results of each training epoch with the verbose set to 1
        # pass the class_weights to use const sensitive learning
        # shuffle the dataset at the end of each training epoch
        # set the callbacks to perform the early stopping
        model.fit (x_train, y_train,
                          batch_size = BATCH_SIZE,
                          epochs = NUMBER_EPOCHS,
                          validation_data = (x_valid, y_valid),
                          verbose = 1,
                          class_weight = class_weights,
                          shuffle = True, 
                          callbacks = EarlyStoppingAtMinLoss (PATIENCE_VALUE, (x_valid, y_valid)))
        # return the trained model
        return model
    print("Dados Selecionados Com sucesso.")

    # Define the model
    # create the KL divergence function
    divergence_fn = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                              tf.cast(x_train_data.shape[0], dtype=tf.float32))
    
    # define the model using the sequential API
    model = Sequential()
    # part 1
    model.add(tfpl.Convolution1DFlipout(
        filters = 96,
        kernel_size = 11,
        strides = 4,
        activation = 'relu',
        input_shape = (SAMPLING_FREQ * 30, 1),
        kernel_prior_fn = tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        kernel_divergence_fn = divergence_fn, 
        bias_prior_fn = tfpl.default_multivariate_normal_fn,
        bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        bias_divergence_fn = divergence_fn
    ))
    model.add(MaxPooling1D(pool_size=3, strides=2))
    # part 2
    model.add(tfpl.Convolution1DFlipout(
        filters = 256,
        kernel_size = 5,
        strides = 1,
        activation = 'relu',
        padding="same",
        kernel_prior_fn = tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        kernel_divergence_fn = divergence_fn, 
        bias_prior_fn = tfpl.default_multivariate_normal_fn,
        bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        bias_divergence_fn = divergence_fn
    ))
    model.add(MaxPooling1D(pool_size=3, strides=2))
    # part 3
    model.add(tfpl.Convolution1DFlipout(
        filters = 384,
        kernel_size = 3,
        strides = 1,
        activation = 'relu',
        padding="same",
        kernel_prior_fn = tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        kernel_divergence_fn = divergence_fn, 
        bias_prior_fn = tfpl.default_multivariate_normal_fn,
        bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        bias_divergence_fn = divergence_fn
    ))
    model.add(tfpl.Convolution1DFlipout(
        filters = 384,
        kernel_size = 3,
        strides = 1,
        activation = 'relu',
        padding="same",
        kernel_prior_fn = tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        kernel_divergence_fn = divergence_fn, 
        bias_prior_fn = tfpl.default_multivariate_normal_fn,
        bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        bias_divergence_fn = divergence_fn
    ))
    model.add(tfpl.Convolution1DFlipout(
        filters = 256,
        kernel_size = 3,
        strides = 1,
        activation = 'relu',
        padding="same",
        kernel_prior_fn = tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        kernel_divergence_fn = divergence_fn, 
        bias_prior_fn = tfpl.default_multivariate_normal_fn,
        bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        bias_divergence_fn = divergence_fn
    ))
    # part 4
    model.add(Flatten())
    model.add(tfpl.DenseFlipout(
        units = 4096, 
        activation = 'relu',
        kernel_prior_fn = tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        kernel_divergence_fn = divergence_fn, 
        bias_prior_fn = tfpl.default_multivariate_normal_fn,
        bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        bias_divergence_fn = divergence_fn
    ))
    model.add(Dropout(0.5))
    model.add(tfpl.DenseFlipout(
        units = 4096, 
        activation = 'relu',
        kernel_prior_fn = tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        kernel_divergence_fn = divergence_fn, 
        bias_prior_fn = tfpl.default_multivariate_normal_fn,
        bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        bias_divergence_fn = divergence_fn
    ))
    model.add(Dropout(0.5))
    model.add(tfpl.DenseFlipout(
        units = 5, 
        activation = 'softmax',
        kernel_prior_fn = tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        kernel_divergence_fn = divergence_fn, 
        bias_prior_fn = tfpl.default_multivariate_normal_fn,
        bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        bias_divergence_fn = divergence_fn
    ))    
    #%% compile, train and test the model
    
    # recommend to change for two fold cross-validation
            
    # compile the model with the ADAM algorithm
    # Keras API automatically adds the Kullback-Leibler divergence (contained on the individual layers of the model) to the cross entropy loss, effectivly computing the ELBO loss
    # track the model accuracy and run witouth the experimental run
    model.compile (optimizer = 'adam', loss = 'categorical_crossentropy',
                  metrics = ['accuracy'], experimental_run_tf_function = False)            
    
    # compute the class wights to be balanced
    class_weights = class_weight.compute_class_weight('balanced',
                                              np.unique(y_train_label),
                                              y_train_label)
    class_weights = {i : class_weights[i] for i in range(classes)}
    
    # the labels are for each 30 s of data so reshape the dataset to have 30 s of data for each label
    # every second of data has 200 points (sampling freqency of 200 Hz) * 30 s of data
    # reshape to the standard: number of examples, number of time steps, number of features for each time step
    x_train_data = x_train_data.reshape (round (len(x_train_data) / (SAMPLING_FREQ * 30)), SAMPLING_FREQ * 30, 1) 
    
    # produce the validation dataset 
    x_train, x_valid, y_train, y_valid = train_test_split (x_train_data, y_train_label, test_size = 0.33, shuffle = True)
    
    # convert the labels to categorical
    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)
    
    # train the model
    model = train_model (x_train, y_train, x_valid, y_valid, model, class_weights)
    
    # Monte Carslo sample the model using the test dataset
    for _ in range(EXAMINE_AVERAGE):
        
        # test the model and return the predictions
        model_prodictions = model.predict (x_test_data.reshape (round (len(x_test_data) / (SAMPLING_FREQ * 30)), SAMPLING_FREQ * 30, 1))
        # convert to a single column array with argmax and estimate the performance metrics    
        m = ConfusionMatrix (actual_vector = np.asarray (y_test_label, dtype = int), predict_vector = np.argmax (model_prodictions, axis=1))      
        Acc_class = m.ACC
        Sen_class = m.TPR
        Spe_class = m.TNR
        AUC_class = m.AUC
        PPV_class = m.PPV
        NPV_class = m.NPV
        
        from pretty_confusion_matrix import pp_matrix_from_data
        cmap = 'copper'
        pp_matrix_from_data(np.asarray (y_test_label, dtype = int), np.argmax (model_prodictions, axis=1) , cmap=cmap)
        plt.savefig("Results\AlexNet1d\ConfusionMatrix.png")
        
        # save the results in a pickle
        f = open("Results\AlexNet1d\ConfusionMatrix.txt", 'ab')
        pickle.dump(m, f)
        f.close()
        # save the results in a pickle
        f = open("Results\AlexNet1d\AccAlexNet1.txt", 'ab')
        pickle.dump(Acc_class, f)
        f.close()
        f = open("Results\AlexNet1d\SenAlexNet1.txt", 'ab')
        pickle.dump(Sen_class, f)
        f.close()
        f = open("Results\AlexNet1d\AlexNet1d\SpeAlexNet1.txt", 'ab')
        pickle.dump(Spe_class, f)
        f.close()
        f = open("Results\AlexNet1d\AucAlexNet1.txt", 'ab')
        pickle.dump(AUC_class, f)
        f.close()
        f = open("Results\AlexNet1d\PpvAlexNet1.txt", 'ab')
        pickle.dump(PPV_class, f)
        f.close()
        f = open(rb"Results\AlexNet1d\NpvAlexNet1.txt", 'ab')
        pickle.dump(NPV_class, f)
        f.close()    
        f = open("Results\AlexNet1d\all_metrics.txt", 'ab')
        pickle.dump(m, f)
        f.close()   
        #model.save("Results\AlexNet1d\model_AlexNet1d")
        print(model_1.summary())


#------------------------------ Train 1 dimension -------------------------

def runmodel_lenet_1d_edit(SAMPLING_FREQ=200, EXAMINE_AVERAGE = 20, THRESHOLD_EARLY_STOPING = 0.005,
                   BATCH_SIZE=32, NUMBER_EPOCHS = 400, PATIENCE_VALUE=40, classes=5):
    print("Train Model is running!")
    print("\nThe parameters that you choose are:")
    print("\Input frequency:", SAMPLING_FREQ) 
    print("Monte Carlo:", EXAMINE_AVERAGE) 
    print("Patience value", THRESHOLD_EARLY_STOPING) 
    print("Batch size:", BATCH_SIZE)  
    print("Maximum number of epoch:", NUMBER_EPOCHS) 
    print("Patience value", PATIENCE_VALUE) 
    print("\nLoading x_train_data.csv - Can take serveral minutes")
    #Load the x_train_data
    x_train_data = np.genfromtxt("x_train_data.csv")
    print("x_train_data.csv loaded sucessfuly")
    print("\nLoading x_test_data.csv - Can take serveral minutes")
    x_test_data = np.genfromtxt("x_test_data.csv")
    print("x_test_data.csv loaded sucessfuly")
    print("\nLoading y_train_label.csv - Can take serveral minutes")
    y_train_label = np.genfromtxt("y_train_label.csv")
    print("y_train_label.csv loaded sucessfuly")
    print("\nLoading y_test_label.csv - Can take serveral minutes")
    y_test_label = np.genfromtxt("y_test_label.csv")
    print("y_test_label.csv loaded sucessfuly")    
 #Define the helping function for the training
    def train_model (x_train, y_train, x_valid, y_valid, model, class_weights):
        # clear the memory
        gc.collect()
        # class to perform the early stopping
        class EarlyStoppingAtMinLoss (tf.keras.callbacks.Callback): 
            # initialization of the class
            def __init__ (self, PATIENCE_VALUE, valid_data): 
                # patience value for the early stopping procedure, defining the maximum number of iteration without an increasse of at least "THRESHOLD_EARLY_STOPING" in the Acc before stopping the training procedure
                super (EarlyStoppingAtMinLoss, self).__init__ ()
                self.patience = PATIENCE_VALUE 
                # best weights of the network
                self.best_weights = None 
                # data used to validate the model
                self.validation_data = valid_data  
    
            # initialize the control parametrers
            def on_train_begin (self, logs = None): 
                # variable holding the number of training iterations without an increasse
                self.wait = 0 
                # variable hold the value of the training epoch where the model early stoped
                self.stopped_epoch = 0 
                # initialization of the variable holding the identified best Acc
                self.best = 0.
                # variable holding the data
                self._data = [] 
                # initialization of the variable holding the Acc of the curent training epoch
                self.curent_Acc = 0.   
    
            # examination at the end of a training epoch
            def on_epoch_end (self, epoch, logs = None): 
                # load the validation data 
                x_val, y_val = self.validation_data [0], self.validation_data [1]   
                # define variable to hold the results with shape: number of examples, number of classes, number of examinations
                probabilistic_output_x_val = np.zeros([len(x_val), 5, EXAMINE_AVERAGE])
                # calculate multiple times the results to assess the epistemic uncertainty 
                for testings in range (0,EXAMINE_AVERAGE,1):
                    # clear GPU memory
                    tf.keras.backend.clear_session()
                    gc.collect()
                    probabilistic_output_x_val[:,:,testings] = model.predict(x_val)
                # estimate the average to determine to remove the epistemic variations from the predictions
                # convert to a single column array with argmax and estimate the accuracy             
                self.curent_Acc = accuracy_score(np.argmax (y_val, axis = 1), np.argmax (np.mean (probabilistic_output_x_val, axis = 2), axis = 1))
                # save the weights if the current Acc is at lest "THRESHOLD_EARLY_STOPING" better than the preivously identifed best Acc
                if np.greater(self.curent_Acc, self.best + THRESHOLD_EARLY_STOPING): 
                    # update the currently best Acc
                    self.best = self.curent_Acc 
                    # restart the counting variable for the early stopping procedure
                    self.wait = 0 
                    # save the weights of the identified best model
                    self.best_weights = self.model.get_weights () 
                else: 
                    # increasse the counting variable for the early stopping procedure
                    self.wait += 1 
                    # early stop the training if the number of training epochs without a minimum Acc increasse of "THRESHOLD_EARLY_STOPING" was higher than the defined patience value
                    if self.wait >= self.patience: 
                        # save the training epoch were the model early stopped
                        self.stopped_epoch = epoch 
                        # flag to identify an early stop
                        self.model.stop_training = True 
                        # restore the weights of the identified best
                        self.model.set_weights (self.best_weights)  
    
            # precedure performed at the end of the training
            def on_train_end (self, logs = None): 
                # report if early stopping occured
                if self.stopped_epoch > 0: 
                    print ('Epoch %05d: early stopping' % (self.stopped_epoch + 1))               
        
        # fit the model using the train dataset, with batch size and number of epochs defined by the user,
        # validate the model at the end of each training epoch with the validation dataset
        # print the results of each training epoch with the verbose set to 1
        # pass the class_weights to use const sensitive learning
        # shuffle the dataset at the end of each training epoch
        # set the callbacks to perform the early stopping
        history = model.fit (x_train, y_train,
                          batch_size = BATCH_SIZE,
                          epochs = NUMBER_EPOCHS,
                          validation_data = (x_valid, y_valid),
                          verbose = 1,
                          class_weight = class_weights,
                          shuffle = True, 
                          callbacks = EarlyStoppingAtMinLoss (PATIENCE_VALUE, (x_valid, y_valid)))
        
        
        # return the trained model
        return (model, history)
    # Define the model
    # Define the model
    # create the KL divergence function
    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                         tf.cast(x_train_data.shape[0], dtype=tf.float32))

  # Define a LeNet-5 model using three convolutional (with max pooling)
  # and two fully connected dense layers. We use the Flipout
  # Monte Carlo estimator for these layers, which enables lower variance
  # stochastic gradients than naive reparameterization.
  
  
  # #convolution
  # number_of_kernels
  # kernel_size
  # stride
  # padding
  
  # #pooling
  # pool_size
  # stride

  # #dense
  # number_of_neurons
  
    model = tf.keras.models.Sequential()
    used_flatten=0
    while True:
          value = eval(input("Last layer (yes=1)(no=0)?"))
          if value:
              number_of_neurons = eval(input("Select the number of output classes? "))
              model.add(
                  tfp.layers.DenseFlipout(
                  number_of_neurons, 
                  kernel_divergence_fn=kl_divergence_function,
                  activation=tf.nn.softmax
                  )
              )
              break
          type_of_layer = eval(input("Select the layer (convolution=0, pooling=1, dense=2)? "))
          if type_of_layer == 0: # convolution
              number_of_kernels = eval(input("Select the number of kernels? "))
              kernel_size = eval(input("Select the kernel size? "))
              strides = eval(input("Select the strides? "))
              padding = eval(input("Select the padding (same=0, valid=1)? "))
              if padding == 0:
                  padding_type = 'SAME'
              else:
                  padding_type = 'VALID'
              model.add(
                  tfp.layers.Convolution1DFlipout(
                      number_of_kernels, 
                      kernel_size=kernel_size,
                      padding=padding_type,
                      strides=strides,
                      kernel_divergence_fn=kl_divergence_function,
                      activation=tf.nn.relu
                  )
              )
          elif type_of_layer == 1: # pooling
              pool_size = eval(input("Select the pool size? "))
              strides = eval(input("Select the strides? "))
              padding = eval(input("Select the padding (same=0, valid=1)? "))
              if padding == 0:
                  padding_type = 'SAME'
              else:
                  padding_type = 'VALID'
              model.add(
                  tf.keras.layers.MaxPooling1D(
                      pool_size=pool_size, 
                      strides=strides, 
                      padding=padding_type
                  )
              )
          else: # dense
              if used_flatten == 0:
                  model.add(Flatten())
                  used_flatten = 1
              number_of_neurons = eval(input("Select the number of neurons? "))
              model.add(
                  tfp.layers.DenseFlipout(
                  number_of_neurons, 
                  kernel_divergence_fn=kl_divergence_function,
                  activation=tf.nn.relu
                  )
              )
    
    #%% compile, train and test the model
    
    # recommend to change for two fold cross-validation
    # compile the model with the ADAM algorithm
    # Keras API automatically adds the Kullback-Leibler divergence (contained on the individual layers of the model) to the cross entropy loss, effectivly computing the ELBO loss
    # track the model accuracy and run witouth the experimental run
    model.compile (optimizer = 'adam', loss = 'categorical_crossentropy',
                  metrics = ['accuracy'], experimental_run_tf_function = False)            
    
    # compute the class wights to be balanced
    class_weights = class_weight.compute_class_weight('balanced',
                                              np.unique(y_train_label),
                                              y_train_label)
    class_weights = {i : class_weights[i] for i in range(classes)}
    
    # the labels are for each 30 s of data so reshape the dataset to have 30 s of data for each label
    # every second of data has 200 points (sampling freqency of 200 Hz) * 30 s of data
    # reshape to the standard: number of examples, number of time steps, number of features for each time step
    x_train_data = x_train_data.reshape (round (len(x_train_data) / (SAMPLING_FREQ * 30)), SAMPLING_FREQ * 30, 1) 
    
    # produce the validation dataset 
    x_train, x_valid, y_train, y_valid = train_test_split (x_train_data, y_train_label, test_size = 0.33, shuffle = True)
    
    # convert the labels to categorical
    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)
    
    # train the model
    model, history = train_model (x_train, y_train, x_valid, y_valid, model, class_weights)



    # Monte Carslo sample the model using the test dataset
    for _ in range(EXAMINE_AVERAGE):
        
        # test the model and return the predictions
        model_prodictions = model.predict (x_test_data.reshape (round (len(x_test_data) / (SAMPLING_FREQ * 30)), SAMPLING_FREQ * 30, 1))
        # convert to a single column array with argmax and estimate the performance metrics    
        m = ConfusionMatrix (actual_vector = np.asarray (y_test_label, dtype = int), predict_vector = np.argmax (model_prodictions, axis=1))
        Acc_class = m.ACC
        Sen_class = m.TPR
        Spe_class = m.TNR
        AUC_class = m.AUC
        PPV_class = m.PPV
        NPV_class = m.NPV
        

        # save the results in a pickle
        f = open("Results\Edit1d\ConfusionMatrix.txt", 'ab')
        pickle.dump(m, f)
        f.close()
        f = open("Results\Edit1d\AccEdit1d.txt", 'ab')
        pickle.dump(Acc_class, f)
        f.close()
        f = open("Results\Edit1d\SenEdit1d.txt", 'ab')
        pickle.dump(Sen_class, f)
        f.close()
        f = open("Results\Edit1d\SpeEdit1d.txt", 'ab')
        pickle.dump(Spe_class, f)
        f.close()
        f = open("Results\Edit1d\AucEdit1d.txt", 'ab')
        pickle.dump(AUC_class, f)
        f.close()
        f = open("Results\Edit1d\PpvEdit1d.txt", 'ab')
        pickle.dump(PPV_class, f)
        f.close()
        f = open(rb"Results\Edit1d\NpvEdit1d.txt", 'ab')
        pickle.dump(NPV_class, f)
        f.close()
        f = open(rb"Results\Edit1d\all_metrics.txt", 'ab')
        pickle.dump(m, f)
        f.close()
    
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid() 
    plt.show()
    plt.savefig("Results\Edit1d\Accuracy.png")           
    
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'validation'], loc='upper left')
    plt.grid()     
    plt.show()
    plt.savefig("Results\Edit1d\Loss.png")    

    from pretty_confusion_matrix import pp_matrix_from_data
    cmap = 'copper'
    pp_matrix_from_data(np.asarray (y_test_label, dtype = int), np.argmax (model_prodictions, axis=1) , cmap=cmap)
    plt.savefig("Results\Edit1d\ConfusionMatrix.png")
    #model.save("Results\Edit1d\model_Edit1d")     
    print(model.summary())




#---------------------  2 dimension LeNet-AlexNet- Train(edit) -----------------

#----------------------  Run 2 dimensions LeNet -----------------------------

def runmodel_lenet_2d(NUM_TRAIN_EXAMPLES = 60000, NUM_HELDOUT_EXAMPLES = 10000, inp1 = 28,
                        inp2 = 28, inp3=1, classes=10,
                      RSV_EPOCHS = 1, DividingValue = 1.33333, CREATE_DATA = 0,
                      data_dir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                             'bayesian_neural_network/data'),
                       model_dir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'bayesian_neural_network/'),
                       viz_steps = 400,
                       fake_data = False, batch_size=16, learning_rate=0.001, num_epochs=300, PATIENCE = 10, num_monte_carlo = 50):
       


    print("Lenet is runnig.")
    print("Epochs update relevant:", learning_rate) 
    print("Batch size:", batch_size) 
    print("Maximum number of epoch:", num_epochs)  
    print("Value for early stopping:", PATIENCE) 
    print("Monte Carlo value:", num_monte_carlo) 
    
     # DO NOT RUN MORE THEN 1, THE MODEL WILL BREAK, RUN ONE BY ONE
     # 2 for 50%; 1.33333 for 25%; 4 for 75%
     # 1 to create new data, otherwise load previous data
     # Directory where data is stored (if using real data)
     # Directory to put the model's fit
     # Frequency at which save visualizations
     # If true, uses fake data. Defaults to real data.
    
    
    AccClass = np.zeros ((RSV_EPOCHS,classes))
    SenClass = np.zeros ((RSV_EPOCHS,classes))
    SpeClass = np.zeros ((RSV_EPOCHS,classes))
    AUCClass = np.zeros ((RSV_EPOCHS,classes))
    PPVClass = np.zeros ((RSV_EPOCHS,classes))
    NPVClass = np.zeros ((RSV_EPOCHS,classes))
    
    
    def create_model():
      """Creates a Keras model using the LeNet-5 architecture.
      Returns:
          model: Compiled Keras model.
      """
      # KL divergence weighted by the number of training samples, using
      # lambda function to pass as input to the kernel_divergence_fn on
      # flipout layers.
      kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))
    
      # Define a LeNet-5 model using three convolutional (with max pooling)
      # and two fully connected dense layers. We use the Flipout
      # Monte Carlo estimator for these layers, which enables lower variance
      # stochastic gradients than naive reparameterization.
      
      model = tf.keras.models.Sequential([
          tfp.layers.Convolution2DFlipout(
              6, kernel_size=5, padding='SAME',
              kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu),
          tf.keras.layers.MaxPooling2D(
              pool_size=[2, 2], strides=[2, 2],
              padding='SAME'),
          tfp.layers.Convolution2DFlipout(
              16, kernel_size=5, padding='SAME',
              kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu),
          tf.keras.layers.MaxPooling2D(
              pool_size=[2, 2], strides=[2, 2],
              padding='SAME'),
          tfp.layers.Convolution2DFlipout(
              120, kernel_size=5, padding='SAME',
              kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu),
          tf.keras.layers.Flatten(),
          tfp.layers.DenseFlipout(
              84, kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu),
          tfp.layers.DenseFlipout(
              classes, kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.softmax)
      ])



      # Model compilation.
      # optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
      
      optimizer = tf.keras.optimizers.Adam()
      
      # We use the categorical_crossentropy loss since the MNIST dataset contains
      # ten labels. The Keras API will then automatically add the
      # Kullback-Leibler divergence (contained on the individual layers of
      # the model), to the cross entropy loss, effectively
      # calcuating the (negated) Evidence Lower Bound Loss (ELBO)
      model.compile(optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'], experimental_run_tf_function=False)
      return model
  
      
    class MNISTSequence(tf.keras.utils.Sequence):
      """Produces a sequence of MNIST digits with labels."""
    
      def __init__(self, data=None, batch_size=batch_size, fake_data_size=None, sample_weights=None):
        """Initializes the sequence.
        Args:
          data: Tuple of numpy `array` instances, the first representing images and
                the second labels.
          batch_size: Integer, number of elements in each training batch.
          fake_data_size: Optional integer number of fake datapoints to generate.
        """
        if data:
          images, labels = data
        else:
          images, labels = MNISTSequence.__generate_fake_data(
              num_images=fake_data_size, num_classes=NUM_CLASSES)
        self.images, self.labels = MNISTSequence.__preprocessing(
            images, labels)
        self.batch_size = batch_size
        self.sample_weights = sample_weights
        
        
        if len(sample_weights) == None:
            sample_weights = np.ones(shape=(len(labels),))
    
      # @staticmethod
      # def __generate_fake_data(num_images, num_classes):
      #   """Generates fake data in the shape of the MNIST dataset for unittest.
      #   Args:
      #     num_images: Integer, the number of fake images to be generated.
      #     num_classes: Integer, the number of classes to be generate.
      #   Returns:
      #     images: Numpy `array` representing the fake image data. The
      #             shape of the array will be (num_images, 28, 28).
      #     labels: Numpy `array` of integers, where each entry will be
      #             assigned a unique integer.
      #   """
      #   images = np.random.randint(low=0, high=256,
      #                              size=(num_images, IMAGE_SHAPE[0],
      #                                    IMAGE_SHAPE[1]))
      #   labels = np.random.randint(low=0, high=num_classes,
      #                              size=num_images)
      #   return images, labels
    
      @staticmethod
      def __preprocessing(images, labels):
        """Preprocesses image and labels data.
        Args:
          images: Numpy `array` representing the image data.
          labels: Numpy `array` representing the labels data (range 0-9).
        Returns:
          images: Numpy `array` representing the image data, normalized
                  and expanded for convolutional network input.
          labels: Numpy `array` representing the labels data (range 0-9),
                  as one-hot (categorical) values.
        """
        images = 2 * (images / 255.) - 1.
        images = images[..., tf.newaxis]
    
        labels = tf.keras.utils.to_categorical(labels)
        return images, labels
    
      def __len__(self):
        return int(tf.math.ceil(len(self.images) / self.batch_size))
    
      def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_sample_weights = self.sample_weights[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y, batch_sample_weights
    
    
    
    
    if tf.io.gfile.exists(model_dir):
      tf.compat.v1.logging.warning(
          'Warning: deleting old log directory at {}'.format(model_dir))
      tf.io.gfile.rmtree(model_dir)
    tf.io.gfile.makedirs(model_dir)
    
    # if fake_data:
    #   train_seq = MNISTSequence(batch_size=batch_size,
    #                             fake_data_size=NUM_TRAIN_EXAMPLES)
    #   heldout_seq = MNISTSequence(batch_size=batch_size,
    #                               fake_data_size=NUM_HELDOUT_EXAMPLES)
    # else:
        
    
      
    train_set, heldout_set = tf.keras.datasets.fashion_mnist.load_data() # for fashion_mnist dataset
    
    x_train = np.array(train_set[0], copy=True)
    y_train = np.array(train_set[1], copy=True)
    
    x_val = x_train[x_train.shape[0]-round(x_train.shape[0]/3):,:,:]
    y_val = y_train[x_train.shape[0]-round(x_train.shape[0]/3):]
    x_train2 = x_train[:x_train.shape[0]-round(x_train.shape[0]/3),:,:]
    y_train2 = y_train[:y_train.shape[0]-round(y_train.shape[0]/3)]
    
    train_seq_train = tuple((x_train2,y_train2)) 
    train_seq_validation = tuple((x_val,y_val))
    
    sample_weights = np.ones(shape=(len(y_train2),))
    
    train_seq_train = MNISTSequence(data=train_seq_train, batch_size=batch_size, sample_weights=sample_weights)
    train_seq_validation = MNISTSequence(data=train_seq_validation, batch_size=batch_size, sample_weights=sample_weights)
    
    sample_weights = np.ones(shape=(len(y_train),))  
    
    train_seq = MNISTSequence(data=train_set, batch_size=batch_size, sample_weights=sample_weights)
    heldout_seq = MNISTSequence(data=heldout_set, batch_size=batch_size, sample_weights=sample_weights)
    
    for RSV_epoch in range (RSV_EPOCHS):
        print ("RSV epoch: " + str (RSV_epoch))
        
        ######################## 1
        print ("Training classifier 1")
        
        model_1 = create_model()
        model_1.build(input_shape=[None, inp1, inp2, inp3])   
        
        best_AUC_macro = 0
        running_patience = 0
        print(' ... Training convolutional neural network')
        
        sample_weight = np.ones(shape=(len(y_train),))
        epoch_accurac_2, epoch_loss_2 = [], []
        epoch_accurac_2_val, epoch_loss_2_val = [], []
        for epoch in range(num_epochs):
            
          epoch_accuracy, epoch_loss = [], []
          
          # for step, (batch_x, batch_y) in enumerate(train_seq):
          for step, (batch_x, batch_y, batch_sample_weights) in enumerate(train_seq_train):
            batch_loss, batch_accuracy = model_1.train_on_batch(
                batch_x, batch_y, sample_weight=batch_sample_weights)
            epoch_accuracy.append(batch_accuracy)
            epoch_loss.append(batch_loss)
        
            if step % 100 == 0:
              print('Iteration: {}, Epoch: {}, Batch index: {}, '
                    'Loss: {:.3f}, Accuracy: {:.3f}'.format(
                        RSV_epoch, epoch, step,
                        tf.reduce_mean(epoch_loss),
                        tf.reduce_mean(epoch_accuracy)))
              
          epoch_accurac_2.append(batch_accuracy)
          epoch_loss_2.append(batch_loss)
          
          # validation check
          probaX=np.zeros([len (train_seq_validation.images),classes,num_monte_carlo])
          probaX[:,:,0] = model_1.predict(train_seq_validation)
          for testings in range (1,num_monte_carlo,1):
              tf.keras.backend.clear_session()
              probaX[:,:,testings]=model_1.predict(train_seq_validation)
          proba=np.mean(probaX, axis=2)
          # probaRetrainVar=np.var(probaX, axis=2)
          # TotalVarRetrain=np.sum(probaRetrainVar, axis=1)
          predictionOneLineC1=np.zeros(len(proba))
          for x in range (len(proba)):
              predictionOneLineC1[x]=np.argmax(proba[x])   
        
          m = ConfusionMatrix(actual_vector= np.argmax(train_seq_validation.labels,axis=1), predict_vector=np.asarray (predictionOneLineC1, dtype=int))
          AUCClassVal = np.zeros(classes) 
          for savingData in range (0, classes, 1):
              AUCClassVal [savingData] = m.AUC [savingData]
              
          AccClassVal = np.zeros(10) 
          for savingData in range (0, classes, 1):
              AccClassVal [savingData] = m.ACC [savingData]
        
        
          #print(m)  
          
          epoch_accurac_2_val.append(np.mean(AccClassVal))
          
          import sklearn
          #A=sklearn.metrics.log_loss(y_true=np.asarray(np.argmax(train_seq_validation.labels,axis=1), dtype=int), y_pred = np.asarray (predictionOneLineC1, dtype=int))
          A=sklearn.metrics.log_loss(y_true=train_seq_validation.labels, y_pred = tf.keras.utils.to_categorical(predictionOneLineC1))
          
          epoch_loss_2_val.append(A)      
        
          if best_AUC_macro + 0.005 < np.mean(AUCClassVal):
              best_AUC_macro = np.mean(AUCClassVal)
              print ("Improved AUC macro to: " + str (best_AUC_macro))
              running_patience = 0
          else:
              running_patience += 1
          if PATIENCE <= running_patience:
              print ("No more improvements, breaking in epoch: " + str (epoch))
              break
     
        probaX=np.zeros([len (heldout_seq.images),classes,num_monte_carlo])
        probaX[:,:,0] = model_1.predict(heldout_seq)
        for testings in range (1,num_monte_carlo,1):
            tf.keras.backend.clear_session()
            probaX[:,:,testings]=model_1.predict(heldout_seq)
        proba10=np.mean(probaX, axis=2)
        predictionOneLineC10=np.zeros(len(proba10))
        for x in range (len(proba10)):
            predictionOneLineC10[x]=np.argmax(proba10[x])  
    
        
        savemat("Results\LeNet2d\proba_PModel_Lenet2d.mat", {'proba': proba})
        f = open("Results\LeNet2d\proba_PModel_Lenet2d.txt", 'ab')
        pickle.dump(proba, f)
        f.close()
              
        m = ConfusionMatrix (actual_vector = heldout_set[1], predict_vector = np.asarray (predictionOneLineC10, dtype = int))

        Acc_class = m.ACC
        Sen_class = m.TPR
        Spe_class = m.TNR
        AUC_class = m.AUC
        PPV_class = m.PPV
        NPV_class = m.NPV
        
        from pretty_confusion_matrix import pp_matrix_from_data
        cmap = 'copper'
        pp_matrix_from_data(heldout_set[1], np.asarray (predictionOneLineC10, dtype = int), cmap=cmap)
        plt.savefig("Results\LeNet2d\Confusion_Matrix.png")
        
        # save the results in a pickle
        f = open("Results\LeNet2d\m1.txt", 'ab')
        pickle.dump(m, f)
        f.close()
        f = open("Results\LeNet2d\AccLeNet2.txt", 'ab')
        pickle.dump(Acc_class, f)
        f.close()
        
        f = open("Results\LeNet2d\SenLeNet2.txt", 'ab')
        pickle.dump(Sen_class, f)
        f.close()
        f = open("Results\LeNet2d\SpeLeNet2.txt", 'ab')
        pickle.dump(Spe_class, f)
        f.close()
        f = open("Results\LeNet2d\AucLeNet2.txt", 'ab')
        pickle.dump(AUC_class, f)
        f.close()
        f = open("Results\LeNet2d\PpvLeNet2.txt", 'ab')
        pickle.dump(PPV_class, f)
        f.close()
        f = open(r"Results\LeNet2d\NpvLeNet2.txt", 'ab')
        pickle.dump(NPV_class, f)
        f.close()   
        f = open(r"Results\LeNet2d\all_metrics.txt", 'ab')
        pickle.dump(m, f)
        f.close() 
        model_1.save("Results\LeNet2d\model_LeNet2d")
        
        
        # summarize history for accuracy
        plt.figure()
        plt.plot(epoch_accurac_2)
        plt.plot(epoch_accurac_2_val)
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.grid()
        plt.show()
        plt.savefig("Results\LeNet2d\prediction.png")
         
        # summarize history for loss
        plt.figure()
        plt.plot(epoch_loss_2)
        plt.plot(epoch_loss_2_val)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.grid()
        plt.show()
        plt.savefig("Results\LeNet2d\prediction.png")
        print(model_1.summary())
   
#-------------------------- Alexnet 2 dimensions     --------------------------------------------------
  
def runmodel_alexnet_2d(NUM_TRAIN_EXAMPLES = 60000, NUM_HELDOUT_EXAMPLES = 10000, inp1 = 28,
                        inp2 = 28, inp3=1, classes=10,
                      RSV_EPOCHS = 1, DividingValue = 1.33333, CREATE_DATA = 0,
                      data_dir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                             'bayesian_neural_network/data'),
                       model_dir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'bayesian_neural_network/'),
                       viz_steps = 400,
                       fake_data = False, batch_size=16, learning_rate=0.001, num_epochs=300, PATIENCE = 10, num_monte_carlo = 50):
    

    print("Lenet is runnig.")
    print("Epochs update relevant:", learning_rate) 
    print("Batch size:", batch_size) 
    print("Maximum number of epoch:", num_epochs)  
    print("Value for early stopping:", PATIENCE) 
    print("Monte Carlo value:", num_monte_carlo) 
     
    AccClass = np.zeros ((RSV_EPOCHS,10))
    SenClass = np.zeros ((RSV_EPOCHS,10))
    SpeClass = np.zeros ((RSV_EPOCHS,10))
    AUCClass = np.zeros ((RSV_EPOCHS,10))
    PPVClass = np.zeros ((RSV_EPOCHS,10))
    NPVClass = np.zeros ((RSV_EPOCHS,10))
    
    
    def create_model():
      """Creates a Keras model using the LeNet-5 architecture.
      Returns:
          model: Compiled Keras model.
      """
      # KL divergence weighted by the number of training samples, using
      # lambda function to pass as input to the kernel_divergence_fn on
      # flipout layers.
      kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))
    
      # Define a LeNet-5 model using three convolutional (with max pooling)
      # and two fully connected dense layers. We use the Flipout
      # Monte Carlo estimator for these layers, which enables lower variance
      # stochastic gradients than naive reparameterization.
      #model = tf.keras.models.Sequential([
    # define the model using the sequential API
    
      model = tf.keras.models.Sequential([
          tfp.layers.Convolution2DFlipout(
              96, kernel_size=11, padding='SAME',
              kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu),
          tf.keras.layers.MaxPooling2D(
              pool_size=[2, 2], strides=[2, 2],
              padding='SAME'),
          tfp.layers.Convolution2DFlipout(
              256, kernel_size=5, padding='SAME',
              kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu),
          tf.keras.layers.MaxPooling2D(
              pool_size=[2, 2], strides=[2, 2],
              padding='SAME'),
          tfp.layers.Convolution2DFlipout(
              384, kernel_size=3, padding='SAME',
              kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu),
          tf.keras.layers.MaxPooling2D(
              pool_size=[2, 2], strides=[2, 2],
              padding='SAME'),
          tfp.layers.Convolution2DFlipout(
              384, kernel_size=3, padding='SAME',
              kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu),
          tf.keras.layers.MaxPooling2D(
              pool_size=[2, 2], strides=[2, 2],
              padding='SAME'),
          tfp.layers.Convolution2DFlipout(
              256, kernel_size=3, padding='SAME',
              kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu),
          tf.keras.layers.Flatten(),
          tfp.layers.DenseFlipout(
              4096, kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu),
          tf.keras.layers.Dropout(0.5),
          tfp.layers.DenseFlipout(
              4096, kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu),
          tf.keras.layers.Dropout(0.5),
          tfp.layers.DenseFlipout(
              classes, kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.softmax)
      ])  


      optimizer = tf.keras.optimizers.Adam() 
      model.compile(optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'], experimental_run_tf_function=False)
      return model
    
    
    class MNISTSequence(tf.keras.utils.Sequence):
      """Produces a sequence of MNIST digits with labels."""
    
      def __init__(self, data=None, batch_size=batch_size, fake_data_size=None, sample_weights=None):
        """Initializes the sequence.
        Args:
          data: Tuple of numpy `array` instances, the first representing images and
                the second labels.
          batch_size: Integer, number of elements in each training batch.
          fake_data_size: Optional integer number of fake datapoints to generate.
        """
        if data:
          images, labels = data
        else:
          images, labels = MNISTSequence.__generate_fake_data(
              num_images=fake_data_size, num_classes=NUM_CLASSES)
        self.images, self.labels = MNISTSequence.__preprocessing(
            images, labels)
        self.batch_size = batch_size
        self.sample_weights = sample_weights
        
        
        if len(sample_weights) == None:
            sample_weights = np.ones(shape=(len(labels),))
     
      @staticmethod
      def __preprocessing(images, labels):
        """Preprocesses image and labels data.
        Args:
          images: Numpy `array` representing the image data.
          labels: Numpy `array` representing the labels data (range 0-9).
        Returns:
          images: Numpy `array` representing the image data, normalized
                  and expanded for convolutional network input.
          labels: Numpy `array` representing the labels data (range 0-9),
                  as one-hot (categorical) values.
        """
        images = 2 * (images / 255.) - 1.
        images = images[..., tf.newaxis]
    
        labels = tf.keras.utils.to_categorical(labels)
        return images, labels
    
      def __len__(self):
        return int(tf.math.ceil(len(self.images) / self.batch_size))
    
      def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_sample_weights = self.sample_weights[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y, batch_sample_weights
    
    
    
    
    if tf.io.gfile.exists(model_dir):
      tf.compat.v1.logging.warning(
          'Warning: deleting old log directory at {}'.format(model_dir))
      tf.io.gfile.rmtree(model_dir)
    tf.io.gfile.makedirs(model_dir)
    
    
    train_set, heldout_set = tf.keras.datasets.fashion_mnist.load_data() # for fashion_mnist dataset
    
    x_train = np.array(train_set[0], copy=True)
    y_train = np.array(train_set[1], copy=True)
    
    x_val = x_train[x_train.shape[0]-round(x_train.shape[0]/3):,:,:]
    y_val = y_train[x_train.shape[0]-round(x_train.shape[0]/3):]
    x_train2 = x_train[:x_train.shape[0]-round(x_train.shape[0]/3),:,:]
    y_train2 = y_train[:y_train.shape[0]-round(y_train.shape[0]/3)]
    
    train_seq_train = tuple((x_train2,y_train2)) 
    train_seq_validation = tuple((x_val,y_val))
    
    sample_weights = np.ones(shape=(len(y_train2),))
    
    train_seq_train = MNISTSequence(data=train_seq_train, batch_size=batch_size, sample_weights=sample_weights)
    train_seq_validation = MNISTSequence(data=train_seq_validation, batch_size=batch_size, sample_weights=sample_weights)
    
    sample_weights = np.ones(shape=(len(y_train),))  
    
    train_seq = MNISTSequence(data=train_set, batch_size=batch_size, sample_weights=sample_weights)
    heldout_seq = MNISTSequence(data=heldout_set, batch_size=batch_size, sample_weights=sample_weights)
    
    
    for RSV_epoch in range (RSV_EPOCHS):
        print ("RSV epoch: " + str (RSV_epoch))
        
        ######################## 1
        print ("Training classifier 1")
        
        model_1 = create_model()
        model_1.build(input_shape=[None, 28, 28, 1])   
        
        best_AUC_macro = 0
        running_patience = 0
        print(' ... Training convolutional neural network')
        
        sample_weight = np.ones(shape=(len(y_train),))
        epoch_accurac_2, epoch_loss_2 = [], []
        epoch_accurac_2_val, epoch_loss_2_val = [], []
        for epoch in range(num_epochs):
            
          epoch_accuracy, epoch_loss = [], []
          
          # for step, (batch_x, batch_y) in enumerate(train_seq):
          for step, (batch_x, batch_y, batch_sample_weights) in enumerate(train_seq_train):
            batch_loss, batch_accuracy = model_1.train_on_batch(
                batch_x, batch_y, sample_weight=batch_sample_weights)
            epoch_accuracy.append(batch_accuracy)
            epoch_loss.append(batch_loss)
        
            if step % 100 == 0:
              print('Iteration: {}, Epoch: {}, Batch index: {}, '
                    'Loss: {:.3f}, Accuracy: {:.3f}'.format(
                        RSV_epoch, epoch, step,
                        tf.reduce_mean(epoch_loss),
                        tf.reduce_mean(epoch_accuracy)))
              
          epoch_accurac_2.append(batch_accuracy)
          epoch_loss_2.append(batch_loss)
          
          # validation check
          probaX=np.zeros([len (train_seq_validation.images),10,num_monte_carlo])
          probaX[:,:,0] = model_1.predict(train_seq_validation)
          for testings in range (1,num_monte_carlo,1):
              tf.keras.backend.clear_session()
              probaX[:,:,testings]=model_1.predict(train_seq_validation)
          proba=np.mean(probaX, axis=2)
          # probaRetrainVar=np.var(probaX, axis=2)
          # TotalVarRetrain=np.sum(probaRetrainVar, axis=1)
          predictionOneLineC1=np.zeros(len(proba))
          for x in range (len(proba)):
              predictionOneLineC1[x]=np.argmax(proba[x])   
        
          m = ConfusionMatrix(actual_vector= np.argmax(train_seq_validation.labels,axis=1), predict_vector=np.asarray (predictionOneLineC1, dtype=int))
          AUCClassVal = np.zeros(10) 
          for savingData in range (0, 10, 1):
              AUCClassVal [savingData] = m.AUC [savingData]
              
          AccClassVal = np.zeros(10) 
          for savingData in range (0, 10, 1):
              AccClassVal [savingData] = m.ACC [savingData]
        
        
          #print(m)  
          
          epoch_accurac_2_val.append(np.mean(AccClassVal))
          
          import sklearn
          #A=sklearn.metrics.log_loss(y_true=np.asarray(np.argmax(train_seq_validation.labels,axis=1), dtype=int), y_pred = np.asarray (predictionOneLineC1, dtype=int))
          A=sklearn.metrics.log_loss(y_true=train_seq_validation.labels, y_pred = tf.keras.utils.to_categorical(predictionOneLineC1))
          
          epoch_loss_2_val.append(A)      
        
          if best_AUC_macro + 0.005 < np.mean(AUCClassVal):
              best_AUC_macro = np.mean(AUCClassVal)
              print ("Improved AUC macro to: " + str (best_AUC_macro))
              running_patience = 0
          else:
              running_patience += 1
          if PATIENCE <= running_patience:
              print ("No more improvements, breaking in epoch: " + str (epoch))
              break
     
        probaX=np.zeros([len (heldout_seq.images),10,num_monte_carlo])
        probaX[:,:,0] = model_1.predict(heldout_seq)
        for testings in range (1,num_monte_carlo,1):
            tf.keras.backend.clear_session()
            probaX[:,:,testings]=model_1.predict(heldout_seq)
        proba10=np.mean(probaX, axis=2)
        predictionOneLineC10=np.zeros(len(proba10))
        for x in range (len(proba10)):
            predictionOneLineC10[x]=np.argmax(proba10[x])  
    
        
        savemat("Results\AlexNet2d\proba_PModel.mat", {'proba': proba})
        f = open("Results\AlexNet2d\proba_PModel.txt", 'ab')
        pickle.dump(proba, f)
        f.close()
        
         
        m = ConfusionMatrix (actual_vector = heldout_set[1], predict_vector = np.asarray (predictionOneLineC10, dtype = int))
        Acc_class = m.ACC
        Sen_class = m.TPR
        Spe_class = m.TNR
        AUC_class = m.AUC
        PPV_class = m.PPV
        NPV_class = m.NPV
        from pretty_confusion_matrix import pp_matrix_from_data
        cmap = 'copper'
        pp_matrix_from_data(heldout_set[1], np.asarray (predictionOneLineC10, dtype = int), cmap=cmap)
        plt.savefig("Results\AlexNet2d\Confusion_Matrix.png")
        
        # save the results in a pickle
        f = open("Results\AlexNet2d\AccAlexNet2.txt", 'ab')
        pickle.dump(Acc_class, f)
        f.close()
        f = open("Results\AlexNet2d\SenAlexNet2.txt", 'ab')
        pickle.dump(Sen_class, f)
        f.close()
        f = open("Results\AlexNet2d\SpeAlexNet2.txt", 'ab')
        pickle.dump(Spe_class, f)
        f.close()
        f = open("Results\AlexNet2d\AucAlexNet2.txt", 'ab')
        pickle.dump(AUC_class, f)
        f.close()
        f = open("Results\AlexNet2d\PpvAlexNet2.txt", 'ab')
        pickle.dump(PPV_class, f)
        f.close()
        f = open(rb"Results\AlexNet2d\NpvAlexNet2.txt", 'ab')
        pickle.dump(NPV_class, f)
        f.close()   
        f = open(r"Results\AlexNet2d\all_metrics.txt", 'ab')
        pickle.dump(m, f)
        f.close() 
        
        # summarize history for accuracy
        plt.figure()
        plt.plot(epoch_accurac_2)
        plt.plot(epoch_accurac_2_val)
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.grid() 
        plt.show()
        plt.savefig("Results\AlexNet2d\prediction.png")
         
        # summarize history for loss
        plt.figure()
        plt.plot(epoch_loss_2)
        plt.plot(epoch_loss_2_val)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.grid() 
        plt.show()
        plt.savefig("Results\AlexNet2d\prediction.png")
        model_1.save("Results\AlexNet2d\model_AlexNet2")
        print(model_1.summary())

        
#-----------------------------  Run Edit 2d         ------------------------------------------------

def runmodel_lenet2_edit(NUM_TRAIN_EXAMPLES = 60000, NUM_HELDOUT_EXAMPLES = 10000, inp1 = 28,
                        inp2 = 28, inp3=1, classes=10,
                      RSV_EPOCHS = 1, DividingValue = 1.33333, CREATE_DATA = 0,
                      data_dir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                             'bayesian_neural_network/data'),
                       model_dir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'bayesian_neural_network/'),
                       viz_steps = 400,
                       fake_data = False, batch_size=16, learning_rate=0.001, num_epochs=300, PATIENCE = 10, num_monte_carlo = 50):
       
    print("Lenet is runnig.")
    print("Epochs update relevant:", learning_rate) 
    print("Batch size:", batch_size) 
    print("Maximum number of epoch:", num_epochs)  
    print("Value for early stopping:", PATIENCE) 
    print("Monte Carlo value:", num_monte_carlo)     


    AccClass = np.zeros ((RSV_EPOCHS,10))
    SenClass = np.zeros ((RSV_EPOCHS,10))
    SpeClass = np.zeros ((RSV_EPOCHS,10))
    AUCClass = np.zeros ((RSV_EPOCHS,10))
    PPVClass = np.zeros ((RSV_EPOCHS,10))
    NPVClass = np.zeros ((RSV_EPOCHS,10))
    
    
    def create_model():
      """Creates a Keras model using the LeNet-5 architecture.
      Returns:
          model: Compiled Keras model.
      """
      # KL divergence weighted by the number of training samples, using
      # lambda function to pass as input to the kernel_divergence_fn on
      # flipout layers.
      kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))
    
      # Define a LeNet-5 model using three convolutional (with max pooling)
      # and two fully connected dense layers. We use the Flipout
      # Monte Carlo estimator for these layers, which enables lower variance
      # stochastic gradients than naive reparameterization.
      
      
      # #convolution
      # number_of_kernels
      # kernel_size
      # stride
      # padding
      
      # #pooling
      # pool_size
      # stride
    
      # #dense
      # number_of_neurons
      
      model = tf.keras.models.Sequential()
      used_flatten=0
      while True:
          value = eval(input("Last layer (yes=1)(no=0)? "))
          if value:
              number_of_neurons = eval(input("Select the number of output classes? "))
              model.add(
                  tfp.layers.DenseFlipout(
                  number_of_neurons, 
                  kernel_divergence_fn=kl_divergence_function,
                  activation=tf.nn.softmax
                  )
              )
              break
          type_of_layer = eval(input("Select the layer (convolution=0, pooling=1, dense=2)? "))
          if type_of_layer == 0: # convolution
              number_of_kernels = eval(input("Select the number of kernels? "))
              kernel_size = eval(input("Select the kernel size? "))
              strides = eval(input("Select the strides? "))
              padding = eval(input("Select the padding (same=0, valid=1)? "))
              if padding == 0:
                  padding_type = 'SAME'
              else:
                  padding_type = 'VALID'
              model.add(
                  tfp.layers.Convolution2DFlipout(
                      number_of_kernels, 
                      kernel_size=kernel_size,
                      padding=padding_type,
                      strides=strides,
                      kernel_divergence_fn=kl_divergence_function,
                      activation=tf.nn.relu
                  )
              )
          elif type_of_layer == 1: # pooling
              pool_size = eval(input("Select the pool size? "))
              strides = eval(input("Select the strides? "))
              padding = eval(input("Select the padding (same=0, valid=1)? "))
              if padding == 0:
                  padding_type = 'SAME'
              else:
                  padding_type = 'VALID'
              model.add(
                  tf.keras.layers.MaxPooling2D(
                      pool_size=[pool_size, pool_size], 
                      strides=[strides, strides],
                      padding=padding_type
                  )
              )
          else: # dense
              if used_flatten == 0:
                  model.add(Flatten())
                  used_flatten = 1
              number_of_neurons = eval(input("Select the number of neurons? "))
              model.add(
                  tfp.layers.DenseFlipout(
                  number_of_neurons, 
                  kernel_divergence_fn=kl_divergence_function,
                  activation=tf.nn.relu
                  )
              )
    
      optimizer = tf.keras.optimizers.Adam()
      
      # We use the categorical_crossentropy loss since the MNIST dataset contains
      # ten labels. The Keras API will then automatically add the
      # Kullback-Leibler divergence (contained on the individual layers of
      # the model), to the cross entropy loss, effectively
      # calcuating the (negated) Evidence Lower Bound Loss (ELBO)
      model.compile(optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'], experimental_run_tf_function=False)
      return model
    
    
    class MNISTSequence(tf.keras.utils.Sequence):
      """Produces a sequence of MNIST digits with labels."""
    
      def __init__(self, data=None, batch_size=batch_size, fake_data_size=None, sample_weights=None):
        """Initializes the sequence.
        Args:
          data: Tuple of numpy `array` instances, the first representing images and
                the second labels.
          batch_size: Integer, number of elements in each training batch.
          fake_data_size: Optional integer number of fake datapoints to generate.
        """
        if data:
          images, labels = data
        else:
          images, labels = MNISTSequence.__generate_fake_data(
              num_images=fake_data_size, num_classes=NUM_CLASSES)
        self.images, self.labels = MNISTSequence.__preprocessing(
            images, labels)
        self.batch_size = batch_size
        self.sample_weights = sample_weights
        
        
        if len(sample_weights) == None:
            sample_weights = np.ones(shape=(len(labels),))
    
    
    
      @staticmethod
      def __preprocessing(images, labels):
        """Preprocesses image and labels data.
        Args:
          images: Numpy `array` representing the image data.
          labels: Numpy `array` representing the labels data (range 0-9).
        Returns:
          images: Numpy `array` representing the image data, normalized
                  and expanded for convolutional network input.
          labels: Numpy `array` representing the labels data (range 0-9),
                  as one-hot (categorical) values.
        """
        images = 2 * (images / 255.) - 1.
        images = images[..., tf.newaxis]
    
        labels = tf.keras.utils.to_categorical(labels)
        return images, labels
    
      def __len__(self):
        return int(tf.math.ceil(len(self.images) / self.batch_size))
    
      def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_sample_weights = self.sample_weights[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y, batch_sample_weights
    
    
    
    
    if tf.io.gfile.exists(model_dir):
      tf.compat.v1.logging.warning(
          'Warning: deleting old log directory at {}'.format(model_dir))
      tf.io.gfile.rmtree(model_dir)
    tf.io.gfile.makedirs(model_dir)
    
    # if fake_data:
    #   train_seq = MNISTSequence(batch_size=batch_size,
    #                             fake_data_size=NUM_TRAIN_EXAMPLES)
    #   heldout_seq = MNISTSequence(batch_size=batch_size
    #                               fake_data_size=NUM_HELDOUT_EXAMPLES)
    # else:
        
    
      
    train_set, heldout_set = tf.keras.datasets.fashion_mnist.load_data() # for fashion_mnist dataset
    
    x_train = np.array(train_set[0], copy=True)
    y_train = np.array(train_set[1], copy=True)
    
    x_val = x_train[x_train.shape[0]-round(x_train.shape[0]/3):,:,:]
    y_val = y_train[x_train.shape[0]-round(x_train.shape[0]/3):]
    x_train2 = x_train[:x_train.shape[0]-round(x_train.shape[0]/3),:,:]
    y_train2 = y_train[:y_train.shape[0]-round(y_train.shape[0]/3)]
    
    train_seq_train = tuple((x_train2,y_train2)) 
    train_seq_validation = tuple((x_val,y_val))
    
    sample_weights = np.ones(shape=(len(y_train2),))
    
    train_seq_train = MNISTSequence(data=train_seq_train, batch_size=batch_size, sample_weights=sample_weights)
    train_seq_validation = MNISTSequence(data=train_seq_validation, batch_size=batch_size, sample_weights=sample_weights)
    
    sample_weights = np.ones(shape=(len(y_train),))  
    
    train_seq = MNISTSequence(data=train_set, batch_size=batch_size, sample_weights=sample_weights)
    heldout_seq = MNISTSequence(data=heldout_set, batch_size=batch_size, sample_weights=sample_weights)
    
    
    
    
    for RSV_epoch in range (RSV_EPOCHS):
        print ("RSV epoch: " + str (RSV_epoch))
        
        ######################## 1
        print ("Training classifier 1")
        
        model_1 = create_model()
        model_1.build(input_shape=[None, 28, 28, 1])   
        
        best_AUC_macro = 0
        running_patience = 0
        print(' ... Training convolutional neural network')
        
        sample_weight = np.ones(shape=(len(y_train),))
        epoch_accurac_2, epoch_loss_2 = [], []
        epoch_accurac_2_val, epoch_loss_2_val = [], []
        for epoch in range(num_epochs):
            
          epoch_accuracy, epoch_loss = [], []
          
          # for step, (batch_x, batch_y) in enumerate(train_seq):
          for step, (batch_x, batch_y, batch_sample_weights) in enumerate(train_seq_train):
            batch_loss, batch_accuracy = model_1.train_on_batch(
                batch_x, batch_y, sample_weight=batch_sample_weights)
            epoch_accuracy.append(batch_accuracy)
            epoch_loss.append(batch_loss)
        
            if step % 100 == 0:
              print('Iteration: {}, Epoch: {}, Batch index: {}, '
                    'Loss: {:.3f}, Accuracy: {:.3f}'.format(
                        RSV_epoch, epoch, step,
                        tf.reduce_mean(epoch_loss),
                        tf.reduce_mean(epoch_accuracy)))
              
          epoch_accurac_2.append(batch_accuracy)
          epoch_loss_2.append(batch_loss)
          
          # validation check
          probaX=np.zeros([len (train_seq_validation.images),10,num_monte_carlo])
          probaX[:,:,0] = model_1.predict(train_seq_validation)
          for testings in range (1,num_monte_carlo,1):
              tf.keras.backend.clear_session()
              probaX[:,:,testings]=model_1.predict(train_seq_validation)
          proba=np.mean(probaX, axis=2)
          # probaRetrainVar=np.var(probaX, axis=2)
          # TotalVarRetrain=np.sum(probaRetrainVar, axis=1)
          predictionOneLineC1=np.zeros(len(proba))
          for x in range (len(proba)):
              predictionOneLineC1[x]=np.argmax(proba[x])   
        
          m = ConfusionMatrix(actual_vector= np.argmax(train_seq_validation.labels,axis=1), predict_vector=np.asarray (predictionOneLineC1, dtype=int))
          AUCClassVal = np.zeros(10) 
          for savingData in range (0, 10, 1):
              AUCClassVal [savingData] = m.AUC [savingData]
              
          AccClassVal = np.zeros(10) 
          for savingData in range (0, 10, 1):
              AccClassVal [savingData] = m.ACC [savingData]
        
        
          #print(m)  
          
          epoch_accurac_2_val.append(np.mean(AccClassVal))
          
          import sklearn
          #A=sklearn.metrics.log_loss(y_true=np.asarray(np.argmax(train_seq_validation.labels,axis=1), dtype=int), y_pred = np.asarray (predictionOneLineC1, dtype=int))
          A=sklearn.metrics.log_loss(y_true=train_seq_validation.labels, y_pred = tf.keras.utils.to_categorical(predictionOneLineC1))
          
          epoch_loss_2_val.append(A)      
        
          if best_AUC_macro + 0.005 < np.mean(AUCClassVal):
              best_AUC_macro = np.mean(AUCClassVal)
              print ("Improved AUC macro to: " + str (best_AUC_macro))
              running_patience = 0
          else:
              running_patience += 1
          if PATIENCE <= running_patience:
              print ("No more improvements, breaking in epoch: " + str (epoch))
              break
     
        probaX=np.zeros([len (heldout_seq.images),10,num_monte_carlo])
        probaX[:,:,0] = model_1.predict(heldout_seq)
        for testings in range (1,num_monte_carlo,1):
            tf.keras.backend.clear_session()
            probaX[:,:,testings]=model_1.predict(heldout_seq)
        proba10=np.mean(probaX, axis=2)
        predictionOneLineC10=np.zeros(len(proba10))
        for x in range (len(proba10)):
            predictionOneLineC10[x]=np.argmax(proba10[x])  

    
        savemat("Results\Edit2d\proba_PModel.mat", {'proba': proba})
    
        f = open("Results\Edit2d\proba_PModel.txt", 'ab')
        pickle.dump(proba, f)
        f.close()
        
        m = ConfusionMatrix (actual_vector = heldout_set[1], predict_vector = np.asarray (predictionOneLineC10, dtype = int))
        Acc_class = m.ACC
        Sen_class = m.TPR
        Spe_class = m.TNR
        AUC_class = m.AUC
        PPV_class = m.PPV
        NPV_class = m.NPV   
        
        from pretty_confusion_matrix import pp_matrix_from_data
        cmap = 'copper'
        pp_matrix_from_data(heldout_set[1], np.asarray (predictionOneLineC10, dtype = int), cmap=cmap)
        plt.savefig("Results\Edit2d\Confusion_Matrix.png")

                # save the results in a pickle
        f = open("Results\Edit2d\AccEdit2d.txt", 'ab')
        pickle.dump(Acc_class, f)
        f.close()
        f = open("Results\Edit2d\SenEdit2d.txt", 'ab')
        pickle.dump(Sen_class, f)
        f.close()
        f = open("Results\Edit2d\SpeEdit2d.txt", 'ab')
        pickle.dump(Spe_class, f)
        f.close()
        f = open("Results\Edit2d\AucEdit2d.txt", 'ab')
        pickle.dump(AUC_class, f)
        f.close()
        f = open("Results\Edit2d\PpvEdit2d.txt", 'ab')
        pickle.dump(PPV_class, f)
        f.close()
        f = open(r"Results\Edit2d\NpvEdit2d.txt", 'ab')
        pickle.dump(NPV_class, f)
        f.close()   
        
        f = open(r"Results\Edit2d\all_metrics.txt", 'ab')
        pickle.dump(m, f)
        f.close() 
        
        # summarize history for accuracy
        plt.figure()
        plt.plot(epoch_accurac_2)
        plt.plot(epoch_accurac_2_val)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.grid() 
        plt.show()
        plt.savefig("Results\Edit2d\prediction.png")
         
        # summarize history for loss
        plt.figure()
        plt.plot(epoch_loss_2)
        plt.plot(epoch_loss_2_val)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.grid() 
        plt.show()
        plt.savefig("Results\Edit2d\prediction.png")
        
        print(model_1.summary())
        model_1.save("Results\Edit2d\model_Edit2d")



#---------------------  Regression LeNet-AlexNet- Train(edit) -----------------

#--------------------- Regression LeNet ---------------------------------------

def run_regression(SAMPLING_FREQ = 100,
                   EXAMINE_AVERAGE = 5,
                   THRESHOLD_EARLY_STOPING = 0.0001,
                   BATCH_SIZE = 32,
                   NUMBER_EPOCHS = 100,
                   PATIENCE_VALUE = 50,
                   classes = 1,
                   inputs = 13
                   ):

    from sklearn.datasets import load_boston
    boston = load_boston()
    x, y = boston.data, boston.target
    print(x.shape)
    
    x = x.reshape(x.shape[0], x.shape[1], 1)
    print(x.shape)
    
    xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.30)
      
    NUM_TRAIN_EXAMPLES = len(x)
    #xtrain = np.expand_dims(xtrain, -1).astype("float32")
    #xtest = np.expand_dims(xtest, -1).astype("float32")
    
    learning_rate = THRESHOLD_EARLY_STOPING
    batch_size = BATCH_SIZE
    num_epochs = NUMBER_EPOCHS
    PATIENCE = PATIENCE_VALUE
    num_monte_carlo = EXAMINE_AVERAGE
    
    def create_model():
      """Creates a Keras model using the LeNet-5 architecture.
      Returns:
          model: Compiled Keras model.
      """
      # KL divergence weighted by the number of training samples, using
      # lambda function to pass as input to the kernel_divergence_fn on
      # flipout layers.
      kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))
    
      # Define a LeNet-5 model using three convolutional (with max pooling)
      # and two fully connected dense layers. We use the Flipout
      # Monte Carlo estimator for these layers, which enables lower variance
      # stochastic gradients than naive reparameterization.
      model = tf.keras.models.Sequential([
          tfp.layers.Convolution1DFlipout(
              6, kernel_size=5, padding='SAME',
              kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu),
          tf.keras.layers.MaxPooling1D(
              pool_size=2, strides=2,
              padding='SAME'),
          tfp.layers.Convolution1DFlipout(
              16, kernel_size=5, padding='SAME',
              kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu),
          tf.keras.layers.MaxPooling1D(
              pool_size=2, strides=2,
              padding='SAME'),
          tfp.layers.Convolution1DFlipout(
              120, kernel_size=5, padding='SAME',
              kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu),
          tf.keras.layers.Flatten(),
          tfp.layers.DenseFlipout(
              84, kernel_divergence_fn=kl_divergence_function,
              activation=tf.nn.relu),
          tfp.layers.DenseFlipout(
              classes, kernel_divergence_fn=kl_divergence_function)
      ])
    
      # Model compilation.
      # optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
      
      optimizer = tf.keras.optimizers.Adam()
      
      # We use the categorical_crossentropy loss since the MNIST dataset contains
      # ten labels. The Keras API will then automatically add the
      # Kullback-Leibler divergence (contained on the individual layers of
      # the model), to the cross entropy loss, effectively
      # calcuating the (negated) Evidence Lower Bound Loss (ELBO)
      model.compile(optimizer="adam", loss='mean_squared_error',
                    metrics=['mean_squared_error'], experimental_run_tf_function=False)
      model.optimizer.learning_rate = learning_rate
      return model

    model = create_model()
    model.build(input_shape=(None,inputs,1))    
       
    import numpy as np 
    from tensorflow import keras
    from matplotlib import pyplot as plt
    from IPython.display import clear_output
    
    
    class PlotLearning(keras.callbacks.Callback):
        """
        Callback to plot the learning curves of the model during training.
        """
        def on_train_begin(self, logs={}):
            self.metrics = {}
            for metric in logs:
                self.metrics[metric] = []
    
        def on_epoch_end(self, epoch, logs={}):
            plt.close()
                    
            # Storing metrics
            for metric in logs:
                if metric in self.metrics:
                    self.metrics[metric].append(logs.get(metric))
                else:
                    self.metrics[metric] = [logs.get(metric)]
            
            # Plotting
            metrics = [x for x in logs if 'val' not in x]
            
            f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
            clear_output(True)
    
            for i, metric in enumerate(metrics):
                axs[i].plot(range(1, epoch + 2), 
                            self.metrics[metric], 
                            label=metric)
                if logs['val_' + metric]:
                    axs[i].plot(range(1, epoch + 2), 
                                self.metrics['val_' + metric], 
                                label='val_' + metric)
                    
                axs[i].legend()
                axs[i].grid()
            plt.tight_layout()
            plt.show()
    
    
    
    x_train, x_valid, y_train, y_valid = train_test_split (xtrain, ytrain, test_size = 0.33, shuffle = True)
           
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=PATIENCE, verbose=0)
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=batch_size,callbacks=[callback], epochs=num_epochs, shuffle=True, verbose=1)
    
    
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('Model Mean Squared Error')
    plt.ylabel('mse')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid()     
    plt.show()
    plt.savefig("Results\RegressionLenet\mse.png")
    
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid() 
    plt.show()
    plt.savefig("Results\RegressionLenet\loss.png")
    
    
    ypred = model.predict(xtest)
    print(model.evaluate(xtrain, ytrain))
    print("MSE: %.4f" % mean_squared_error(ytest, ypred))
    
    # verify the model performance
    #model.evaluate(xtest, ytest, verbose=1)
    
    num_monte_carlo = 10
    # validation check
    probaX=np.zeros([len (xtest),1,num_monte_carlo])
    probaX[:,:,0] = model.predict(xtest)
    for testings in range (1,num_monte_carlo,1):
        tf.keras.backend.clear_session()
        probaX[:,:,testings]=model.predict(xtest)
    proba=np.mean(probaX, axis=2)
    mse = mean_squared_error(ytest, proba)
    print("MSE for mean: %.4f" % mse)
    
    f = open("Results\RegressionLenet\MSE2d.txt", 'ab')
    pickle.dump(mse, f)
    f.close()
    
    savemat("Results\RegressionLenet\proba_PModel.mat", {'proba': proba})
    #Média das amostras todas de monte Carlo
    f = open("Results\RegressionLenet\proba_PModel.txt", 'ab')
    pickle.dump(proba, f)
    f.close()
    
    savemat("Results\RegressionLenet\probaX_PModel.mat", {'probaX': probaX})
    #Anmostras todas de monte carlo
    f = open("Results\RegressionLenet\probaX_PModel.txt", 'ab')
    pickle.dump(probaX, f)
    f.close()
    
    plt.figure()
    x_ax = range(len(ypred))
    plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
    plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.grid() 
    plt.show()
    plt.savefig("Results\RegressionLenet\prediction.png")
    
    print(model.summary())    
    model.save("Results\RegressionLenet\model_Regression")

    
 
    

    
#-------------------------- AlexNet for Regression -----------------------------    

def run_regression_edit1(SAMPLING_FREQ = 100,
                   EXAMINE_AVERAGE = 5,
                   THRESHOLD_EARLY_STOPING = 0.0001,
                   BATCH_SIZE = 32,
                   NUMBER_EPOCHS = 100,
                   PATIENCE_VALUE = 50,
                   classes = 1,
                   inputs = 13
                   ):

    from sklearn.datasets import load_boston
    boston = load_boston()
    x, y = boston.data, boston.target
    print(x.shape)
    
    x = x.reshape(x.shape[0], x.shape[1], 1)
    print(x.shape)
    
    xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.30)
      
    NUM_TRAIN_EXAMPLES = len(x)
    #xtrain = np.expand_dims(xtrain, -1).astype("float32")
    #xtest = np.expand_dims(xtest, -1).astype("float32")
    
    learning_rate = THRESHOLD_EARLY_STOPING
    batch_size = BATCH_SIZE
    num_epochs = NUMBER_EPOCHS
    PATIENCE = PATIENCE_VALUE
    num_monte_carlo = EXAMINE_AVERAGE
    
    def create_model():
      """Creates a Keras model using the LeNet-5 architecture.
      Returns:
          model: Compiled Keras model.
      """
      # KL divergence weighted by the number of training samples, using
      # lambda function to pass as input to the kernel_divergence_fn on
      # flipout layers.
      kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))
    
      # Define a LeNet-5 model using three convolutional (with max pooling)
      # and two fully connected dense layers. We use the Flipout
      # Monte Carlo estimator for these layers, which enables lower variance
      # stochastic gradients than naive reparameterization.
      
      model = tf.keras.models.Sequential([
        tfp.layers.Convolution1DFlipout(
            96, kernel_size=11, padding='SAME',
            kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.relu),
        tf.keras.layers.MaxPooling1D(
            pool_size=2, strides=2,
            padding='SAME'),
        tfp.layers.Convolution1DFlipout(
            256, kernel_size=5, padding='SAME',
            kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.relu),
        tf.keras.layers.MaxPooling1D(
            pool_size=2, strides=2,
            padding='SAME'),
        tfp.layers.Convolution1DFlipout(
            384, kernel_size=3, padding='SAME',
            kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.relu),
        tf.keras.layers.MaxPooling1D(
            pool_size=2, strides=2,
            padding='SAME'),
        tfp.layers.Convolution1DFlipout(
            384, kernel_size=3, padding='SAME',
            kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.relu),
        tf.keras.layers.MaxPooling1D(
            pool_size=2, strides=2,
            padding='SAME'),
        tfp.layers.Convolution1DFlipout(
            256, kernel_size=3, padding='SAME',
            kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        tfp.layers.DenseFlipout(
            4096, kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tfp.layers.DenseFlipout(
            4096, kernel_divergence_fn=kl_divergence_function,
            activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tfp.layers.DenseFlipout(
              classes, kernel_divergence_fn=kl_divergence_function)
      ])
      # Model compilation.
      # optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
      
      optimizer = tf.keras.optimizers.Adam()
      
      # We use the categorical_crossentropy loss since the MNIST dataset contains
      # ten labels. The Keras API will then automatically add the
      # Kullback-Leibler divergence (contained on the individual layers of
      # the model), to the cross entropy loss, effectively
      # calcuating the (negated) Evidence Lower Bound Loss (ELBO)
      model.compile(optimizer="adam", loss='mean_squared_error',
                    metrics=['mean_squared_error'], experimental_run_tf_function=False)
      model.optimizer.learning_rate = learning_rate
      return model

    model = create_model()
    model.build(input_shape=(None,inputs,1))    
       
    import numpy as np 
    from tensorflow import keras
    from matplotlib import pyplot as plt
    from IPython.display import clear_output
    
    
    class PlotLearning(keras.callbacks.Callback):
        """
        Callback to plot the learning curves of the model during training.
        """
        def on_train_begin(self, logs={}):
            self.metrics = {}
            for metric in logs:
                self.metrics[metric] = []
    
        def on_epoch_end(self, epoch, logs={}):
            plt.close()
                    
            # Storing metrics
            for metric in logs:
                if metric in self.metrics:
                    self.metrics[metric].append(logs.get(metric))
                else:
                    self.metrics[metric] = [logs.get(metric)]
            
            # Plotting
            metrics = [x for x in logs if 'val' not in x]
            
            f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
            clear_output(True)
    
            for i, metric in enumerate(metrics):
                axs[i].plot(range(1, epoch + 2), 
                            self.metrics[metric], 
                            label=metric)
                if logs['val_' + metric]:
                    axs[i].plot(range(1, epoch + 2), 
                                self.metrics['val_' + metric], 
                                label='val_' + metric)
                    
                axs[i].legend()
                axs[i].grid()
            plt.tight_layout()
            plt.show()
    
    
    
    x_train, x_valid, y_train, y_valid = train_test_split (xtrain, ytrain, test_size = 0.33, shuffle = True)
           
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=PATIENCE, verbose=0)
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=batch_size,callbacks=[callback], epochs=num_epochs, shuffle=True, verbose=1)
    
    
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('Model mse')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid() 
    plt.show()
    plt.savefig("Results\RegressionAlexNet\mse.png")
    
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid() 
    plt.show()
    plt.savefig("Results\RegressionAlexNet\loss.png")
    
    
    ypred = model.predict(xtest)
    print(model.evaluate(xtrain, ytrain))
    print("MSE: %.4f" % mean_squared_error(ytest, ypred))
    
    # verify the model performance
    #model.evaluate(xtest, ytest, verbose=1)
    
    num_monte_carlo = 10
    # validation check
    probaX=np.zeros([len (xtest),1,num_monte_carlo])
    probaX[:,:,0] = model.predict(xtest)
    for testings in range (1,num_monte_carlo,1):
        tf.keras.backend.clear_session()
        probaX[:,:,testings]=model.predict(xtest)
    proba=np.mean(probaX, axis=2)
    mse = mean_squared_error(ytest, proba)
    print("MSE for mean: %.4f" % mse)
    
    f = open("Results\RegressionAlexNet\MSE2d.txt", 'ab')
    pickle.dump(mse, f)
    f.close()
    
    savemat("Results\RegressionAlexNet\proba_PModel.mat", {'proba': proba})
    #Média das amostras todas de monte Carlo
    f = open("Results\RegressionAlexNet\proba_PModel.txt", 'ab')
    pickle.dump(proba, f)
    f.close()
    
    savemat("Results\RegressionAlexNet\probaX_PModel.mat", {'probaX': probaX})
    #Anmostras todas de monte carlo
    f = open("Results\RegressionAlexNet\probaX_PModel.txt", 'ab')
    pickle.dump(probaX, f)
    f.close()
    
    plt.figure()
    x_ax = range(len(ypred))
    plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
    plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.grid() 
    plt.show()
    plt.savefig("Results\RegressionAlexNet\prediction.png")  
    model.save("Results\RegressionAlexNet\model_Regression")
    print(model.summary())

    
        
#------------------------------- Regression edit -----------------------------------        
        
def run_regression_edit(SAMPLING_FREQ = 100,
                   EXAMINE_AVERAGE = 5,
                   THRESHOLD_EARLY_STOPING = 0.0001,
                   BATCH_SIZE = 32,
                   NUMBER_EPOCHS = 100,
                   PATIENCE_VALUE = 50,
                   classes = 1,
                   inputs = 13
                   ):

    from sklearn.datasets import load_boston
    boston = load_boston()
    x, y = boston.data, boston.target
    print(x.shape)
    
    x = x.reshape(x.shape[0], x.shape[1], 1)
    print(x.shape)
    
    xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.30)
      
    NUM_TRAIN_EXAMPLES = len(x)
    #xtrain = np.expand_dims(xtrain, -1).astype("float32")
    #xtest = np.expand_dims(xtest, -1).astype("float32")
    
    learning_rate = THRESHOLD_EARLY_STOPING
    batch_size = BATCH_SIZE
    num_epochs = NUMBER_EPOCHS
    PATIENCE = PATIENCE_VALUE
    num_monte_carlo = EXAMINE_AVERAGE
    
    
    def create_model():
      """Creates a Keras model using the LeNet-5 architecture.
      Returns:
          model: Compiled Keras model.
      """
      # KL divergence weighted by the number of training samples, using
      # lambda function to pass as input to the kernel_divergence_fn on
      # flipout layers.
      kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))
    
      # Define a LeNet-5 model using three convolutional (with max pooling)
      # and two fully connected dense layers. We use the Flipout
      # Monte Carlo estimator for these layers, which enables lower variance
      # stochastic gradients than naive reparameterization.
      model = tf.keras.models.Sequential()
      used_flatten=0
      while True:
          value = eval(input("Last layer (yes=1)(no=0)? "))
          if value:
              number_of_neurons = eval(input("Select the number of output classes? "))
              model.add(
                  tfp.layers.DenseFlipout(
                  number_of_neurons, 
                  kernel_divergence_fn=kl_divergence_function,
#                  activation=tf.nn.softmax
                  )
              )
              break
          type_of_layer = eval(input("Select the layer (convolution=0, pooling=1, dense=2)? "))
          if type_of_layer == 0: # convolution
              number_of_kernels = eval(input("Select the number of kernels? "))
              kernel_size = eval(input("Select the kernel size? "))
              strides = eval(input("Select the strides? "))
              padding = eval(input("Select the padding (same=0, valid=1)? "))
              if padding == 0:
                  padding_type = 'SAME'
              else:
                  padding_type = 'VALID'
              model.add(
                  tfp.layers.Convolution1DFlipout(
                      number_of_kernels, 
                      kernel_size=kernel_size,
                      padding=padding_type,
                      strides=strides,
                      kernel_divergence_fn=kl_divergence_function,
                      activation=tf.nn.relu
                  )
              )
          elif type_of_layer == 1: # pooling
              pool_size = eval(input("Select the pool size? "))
              strides = eval(input("Select the strides? "))
              padding = eval(input("Select the padding (same=0, valid=1)? "))
              if padding == 0:
                  padding_type = 'SAME'
              else:
                  padding_type = 'VALID'
              model.add(
                  tf.keras.layers.MaxPooling1D(
                      pool_size=pool_size, 
                      strides=strides,
                      padding=padding_type
                  )
              )
          else: # dense
              if used_flatten == 0:
                  model.add(Flatten())
                  used_flatten = 1
              number_of_neurons = eval(input("Select the number of neurons? "))
              model.add(
                  tfp.layers.DenseFlipout(
                  number_of_neurons, 
                  kernel_divergence_fn=kl_divergence_function,
                  activation=tf.nn.relu
                  )
              )
    
    
      # Model compilation.
      # optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
      
      optimizer = tf.keras.optimizers.Adam()
      
      # We use the categorical_crossentropy loss since the MNIST dataset contains
      # ten labels. The Keras API will then automatically add the
      # Kullback-Leibler divergence (contained on the individual layers of
      # the model), to the cross entropy loss, effectively
      # calcuating the (negated) Evidence Lower Bound Loss (ELBO)
      model.compile(optimizer="adam", loss='mean_squared_error',
                    metrics=['mean_squared_error'], experimental_run_tf_function=False)
      model.optimizer.learning_rate = learning_rate
      return model

    model = create_model()
    model.build(input_shape=(None,inputs,1))    
       
    import numpy as np 
    from tensorflow import keras
    from matplotlib import pyplot as plt
    from IPython.display import clear_output
    
    
    class PlotLearning(keras.callbacks.Callback):
        """
        Callback to plot the learning curves of the model during training.
        """
        def on_train_begin(self, logs={}):
            self.metrics = {}
            for metric in logs:
                self.metrics[metric] = []
    
        def on_epoch_end(self, epoch, logs={}):
            plt.close()
                    
            # Storing metrics
            for metric in logs:
                if metric in self.metrics:
                    self.metrics[metric].append(logs.get(metric))
                else:
                    self.metrics[metric] = [logs.get(metric)]
            
            # Plotting
            metrics = [x for x in logs if 'val' not in x]
            
            f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
            clear_output(True)
    
            for i, metric in enumerate(metrics):
                axs[i].plot(range(1, epoch + 2), 
                            self.metrics[metric], 
                            label=metric)
                if logs['val_' + metric]:
                    axs[i].plot(range(1, epoch + 2), 
                                self.metrics['val_' + metric], 
                                label='val_' + metric)
                    
                axs[i].legend()
                axs[i].grid()
            plt.tight_layout()
            plt.show()
    
    
    
    x_train, x_valid, y_train, y_valid = train_test_split (xtrain, ytrain, test_size = 0.33, shuffle = True)
           
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=PATIENCE, verbose=0)
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=batch_size,callbacks=[callback], epochs=num_epochs, shuffle=True, verbose=1)
    
    
    print(history.history.keys())
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('Model Mse')
    plt.ylabel('Mse')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid() 
    plt.show()
    plt.savefig("Results\RegressionEdit\Mse.png")
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid() 
    plt.show()
    plt.savefig("Results\RegressionEdit\loss.png") 
    
    ypred = model.predict(xtest)
    print(model.evaluate(xtrain, ytrain))
    print("MSE: %.4f" % mean_squared_error(ytest, ypred))
    
    # verify the model performance
    #model.evaluate(xtest, ytest, verbose=1)
    
    # validation check
    probaX=np.zeros([len (xtest),1,num_monte_carlo])
    probaX[:,:,0] = model.predict(xtest)
    for testings in range (1,num_monte_carlo,1):
        tf.keras.backend.clear_session()
        probaX[:,:,testings]=model.predict(xtest)
    proba=np.mean(probaX, axis=2)
    mse = mean_squared_error(ytest, proba)
    print("MSE for mean: %.4f" % mse)
    
    f = open("Results\RegressionEdit\MSE2d.txt", 'ab')
    pickle.dump(mse, f)
    f.close()
    
    savemat("Results\RegressionEdit\proba_PModel.mat", {'proba': proba})
    #Média das amostras todas de monte Carlo
    f = open("Results\RegressionEdit\proba_PModel.txt", 'ab')
    pickle.dump(proba, f)
    f.close()
    
    savemat("Results\RegressionEdit\probaX_PModel.mat", {'probaX': probaX})
    #Anmostras todas de monte carlo
    f = open("Results\RegressionEdit\probaX_PModel.txt", 'ab')
    pickle.dump(probaX, f)
    f.close()
    
    plt.figure()
    x_ax = range(len(ypred))
    plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
    plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.grid() 
    plt.show()
    plt.savefig("Results\RegressionEdit\Regression.png")
    print(model.summary())
    model.save("Results\RegressionEdit\model_Regression")
