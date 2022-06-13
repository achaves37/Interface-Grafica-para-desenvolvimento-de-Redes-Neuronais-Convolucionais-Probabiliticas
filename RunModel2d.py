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
tfd = tfp.distributions
tfpl = tfp.layers

#----------------------    Funções --------------------- 
 
   
#-------------------------- Lenet MNIST --------------------------------------------------
def runmodel_lenet_2d(IMAGE_SHAPE = [28, 28, 1], NUM_TRAIN_EXAMPLES = 60000, NUM_HELDOUT_EXAMPLES = 10000,
                      NUM_CLASSES = 10, RSV_EPOCHS = 1, DividingValue = 1.33333, CREATE_DATA = 0,
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
              NUM_CLASSES, kernel_divergence_fn=kl_divergence_function,
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
        model_1.build(input_shape=[None, 28, 28, 1])
        
        best_AUC_macro = 0
        running_patience = 0
        print(' ... Training convolutional neural network')
        
        sample_weight = np.ones(shape=(len(y_train),))
        
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
    
        
        savemat("LeNet2d\proba_PModel_Lenet2d.mat", {'proba': proba})
    
        f = open("LeNet2d\proba_PModel_Lenet2d.txt", 'ab')
        pickle.dump(proba, f)
        f.close()
    
        
        m = ConfusionMatrix (actual_vector = heldout_set[1], predict_vector = np.asarray (predictionOneLineC10, dtype = int))
        Acc_class = m.ACC
        Sen_class = m.TPR
        Spe_class = m.TNR
        AUC_class = m.AUC
        PPV_class = m.PPV
        NPV_class = m.NPV
        
        # save the results in a pickle
        f = open("LeNet2d\AccLenet2.txt", 'ab')
        pickle.dump(Acc_class, f)
        f.close()
        f = open("LeNet2d\SenLenet2.txt", 'ab')
        pickle.dump(Sen_class, f)
        f.close()
        f = open("LeNet2d\SpeLenet2.txt", 'ab')
        pickle.dump(Spe_class, f)
        f.close()
        f = open("LeNet2d\AucLenet2.txt", 'ab')
        pickle.dump(AUC_class, f)
        f.close()
        f = open("LeNet2d\PpvLenet2.txt", 'ab')
        pickle.dump(PPV_class, f)
        f.close()
        f = open(r"LeNet2d\NpvLenet2.txt", 'ab')
        pickle.dump(NPV_class, f)
        f.close()   
        f = open(r"LeNet2d\all_metrics.txt", 'ab')
        pickle.dump(m, f)
        f.close() 
        
        model_1("LeNet2d\model_LeNet2d").save
        print("Lenet is finish go to the tab results.")

#-------------------------- Lenet MNIST --------------------------------------------------

        
def runmodel_alexnet_2d(IMAGE_SHAPE = [28, 28, 1], NUM_TRAIN_EXAMPLES = 60000, NUM_HELDOUT_EXAMPLES = 10000,
                      NUM_CLASSES = 10, RSV_EPOCHS = 1, DividingValue = 1.33333, CREATE_DATA = 0,
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
              NUM_CLASSES, kernel_divergence_fn=kl_divergence_function,
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
    
        
        savemat("proba_PModel.mat", {'proba': proba})
    
        f = open("proba_PModel.txt", 'ab')
        pickle.dump(proba, f)
        f.close()
         
        m = ConfusionMatrix (actual_vector = heldout_set[1], predict_vector = np.asarray (predictionOneLineC10, dtype = int))
        Acc_class = m.ACC
        Sen_class = m.TPR
        Spe_class = m.TNR
        AUC_class = m.AUC
        PPV_class = m.PPV
        NPV_class = m.NPV
        
        # save the results in a pickle
        f = open("AlexNet2d\AccAlexNet2.txt", 'ab')
        pickle.dump(Acc_class, f)
        f.close()
        f = open("AlexNet2d\SenAlexNet2.txt", 'ab')
        pickle.dump(Sen_class, f)
        f.close()
        f = open("AlexNet2d\SpeAlexNet2.txt", 'ab')
        pickle.dump(Spe_class, f)
        f.close()
        f = open("AlexNet2d\AucAlexNet2.txt", 'ab')
        pickle.dump(AUC_class, f)
        f.close()
        f = open("AlexNet2d\PpvAlexNet2.txt", 'ab')
        pickle.dump(PPV_class, f)
        f.close()
        f = open(rb"AlexNet2d\NpvAlexNet2.txt", 'ab')
        pickle.dump(NPV_class, f)
        f.close()   
        f = open(r"AlexNet2d\all_metrics.txt", 'ab')
        pickle.dump(m, f)
        f.close()      
        print("Lenet is finish go to the tab results.")
        
        
        
