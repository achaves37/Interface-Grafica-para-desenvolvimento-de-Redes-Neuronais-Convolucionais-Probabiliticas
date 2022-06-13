from tkinter import *
from tkinter import ttk
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
tfd = tfp.distributions
tfpl = tfp.layers
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

#-------------------------- Modelo 1 lenet IRCU SLEEP -----------------------------------------

def runmodel_lenet_1d(SAMPLING_FREQ=200, EXAMINE_AVERAGE = 20, THRESHOLD_EARLY_STOPING = 0.005,
                   BATCH_SIZE=32, NUMBER_EPOCHS = 400, PATIENCE_VALUE=40):
    print("LeNet Model is running!")
    print("\nThe parameters that you choose are:")
    print("\nSignals sampling frequency:", SAMPLING_FREQ) 
    print("Times shoul the algoritm run:", EXAMINE_AVERAGE) 
    print("Epochs update relevant:", THRESHOLD_EARLY_STOPING) 
    print("Batch size:", BATCH_SIZE)  
    print("Maximum number of epoch:", NUMBER_EPOCHS) 
    print("Value for early stopping:", PATIENCE_VALUE) 
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
    # Define the model
    # create the KL divergence function
    divergence_fn = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                              tf.cast(x_train_data.shape[0], dtype=tf.float32))
    
    # define the model using the sequential API
    model = Sequential()
    model.add(tfpl.Convolution1DFlipout(
        filters = 128,
        kernel_size = 7,
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
        filters = 128,
        kernel_size = 7,
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
        units = 100, 
        activation = 'relu',
        kernel_prior_fn = tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        kernel_divergence_fn = divergence_fn, 
        bias_prior_fn = tfpl.default_multivariate_normal_fn,
        bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular = False),
        bias_divergence_fn = divergence_fn
    ))
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
    class_weights = {i : class_weights[i] for i in range(5)}
    
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
        # save the results in a pickle
        f = open("LeNet1d\AccLenet1.txt", 'ab')
        pickle.dump(Acc_class, f)
        f.close()
        f = open("LeNet1d\SenLenet1.txt", 'ab')
        pickle.dump(Sen_class, f)
        f.close()
        f = open("LeNet1d\SpeLenet1.txt", 'ab')
        pickle.dump(Spe_class, f)
        f.close()
        f = open("LeNet1d\AucLenet1.txt", 'ab')
        pickle.dump(AUC_class, f)
        f.close()
        f = open("LeNet1d\PpvLenet1.txt", 'ab')
        pickle.dump(PPV_class, f)
        f.close()
        f = open(rb"LeNet1d\NpvLenet1.txt", 'ab')
        pickle.dump(NPV_class, f)
        f.close()
        f = open(rb"LeNet1d\all_metrics.txt", 'ab')
        pickle.dump(m, f)
        f.close()   


#-------------------------- Modelo 1 IRCU SLEEP alexlenet -----------------------------------------

def runmodel_alexnet():
    
    print("Alexnet Model is running!")
    print("\nThe parameters that you choose are:")
    print("\nSignals sampling frequency:", SAMPLING_FREQ) 
    print("Times shoul the algoritm run:", EXAMINE_AVERAGE) 
    print("Epochs update relevant:", THRESHOLD_EARLY_STOPING) 
    print("Batch size:", BATCH_SIZE)  
    print("Maximum number of epoch:", NUMBER_EPOCHS) 
    print("Value for early stopping:", PATIENCE_VALUE) 
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
    class_weights = {i : class_weights[i] for i in range(5)}
    
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
        # save the results in a pickle
        f = open("AlexNet1d\AccAlexNet1.txt", 'ab')
        pickle.dump(Acc_class, f)
        f.close()
        f = open("AlexNet1d\SenAlexNet1.txt", 'ab')
        pickle.dump(Sen_class, f)
        f.close()
        f = open("AlexNet1d\AlexNet1d\SpeAlexNet1.txt", 'ab')
        pickle.dump(Spe_class, f)
        f.close()
        f = open("AlexNet1d\AucAlexNet1.txt", 'ab')
        pickle.dump(AUC_class, f)
        f.close()
        f = open("AlexNet1d\PpvAlexNet1.txt", 'ab')
        pickle.dump(PPV_class, f)
        f.close()
        f = open(rb"AlexNet1d\NpvAlexNet1.txt", 'ab')
        pickle.dump(NPV_class, f)
        f.close()    
        f = open("AlexNet1d\all_metrics.txt", 'ab')
        pickle.dump(m, f)
        f.close()   
        
        
        
        
        

