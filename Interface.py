from tkinter import *
from tkinter import ttk
import sys
import os
from os import path
import tensorflow_probability as tfp
import tensorflow as tf
import tensorflow.compat.v2 as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D, Dropout, Flatten, Conv2D, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
import scipy.io as spio
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pycm import *
# import the garbage colector to clear the memory
import gc
# importing the library to save the results
import pickle
# importing the library to check the dataset files
import csv
import matplotlib.pyplot as plt
import warnings
import numpy as np
from scipy.io import savemat
from pycm import *
from Results import *
from RunModel import *
from sklearn.metrics import mean_squared_error
tfd = tfp.distributions
tfpl = tfp.layers
import webbrowser


#-------------------------- Saves parameters --------------------------------   

#------------------ Save parameters 1 dimension Lenet-AlexNet-Edit-------------
def runmodel_lenet_1d_call():
    runmodel_lenet_1d(SAMPLING_FREQ = int(texto_ins1.get()),
                       EXAMINE_AVERAGE= int(texto_ins2.get()),
                       THRESHOLD_EARLY_STOPING = float(texto_ins4.get()),
                       BATCH_SIZE = int(texto_ins5.get()),
                       NUMBER_EPOCHS = int(texto_ins6.get()),
                       PATIENCE_VALUE = int(texto_ins7.get()),
                       classes = int(texto_ins8.get())
                       )
    return
def alexnet_resultado_generic1_call():
    runmodel_alexnet(SAMPLING_FREQ = int(texto_ins1.get()),
                       EXAMINE_AVERAGE= int(texto_ins2.get()),
                       THRESHOLD_EARLY_STOPING = float(texto_ins4.get()),
                       BATCH_SIZE = int(texto_ins5.get()),
                       NUMBER_EPOCHS = int(texto_ins6.get()),
                       PATIENCE_VALUE = int(texto_ins7.get()),
                       classes = int(texto_ins8.get())
                       )
    return
def runmodel_edit_1d_call():
    runmodel_lenet_1d_edit(SAMPLING_FREQ = int(texto_ins1.get()),
                       EXAMINE_AVERAGE= int(texto_ins2.get()),
                       THRESHOLD_EARLY_STOPING = float(texto_ins4.get()),
                       BATCH_SIZE = int(texto_ins5.get()),
                       NUMBER_EPOCHS = int(texto_ins6.get()),
                       PATIENCE_VALUE = int(texto_ins7.get()),
                       classes = int(texto_ins8.get())
                       )
    return


#------------------ Save parameters 2 dimensions Lenet-AlexNet-Edit-----------
def runmodel_lenet_2d_call():
    runmodel_lenet_2d(learning_rate = float(texto_ins22d.get()),
                       batch_size = int(texto_ins42d.get()),
                       num_epochs = int(texto_ins52d.get()),
                       PATIENCE = int(texto_ins62d.get()),
                       num_monte_carlo = int(texto_ins72d.get()),
                       inp1 = int(texto_ins82d.get()),
                       inp2 = int(texto_ins92d.get()),
                       inp3 = int(texto_ins02d.get()),
                       classes = int(texto_ins02d1.get())
                       )  
    return
def runmodel_alexlenet_2d_call():
    runmodel_alexnet_2d(learning_rate = float(texto_ins22d.get()),
                       batch_size = int(texto_ins42d.get()),
                       num_epochs = int(texto_ins52d.get()),
                       PATIENCE = int(texto_ins62d.get()),
                       num_monte_carlo = int(texto_ins72d.get()),
                       inp1 = int(texto_ins82d.get()),
                       inp2 = int(texto_ins92d.get()),
                       inp3 = int(texto_ins02d.get()),
                       classes = int(texto_ins02d1.get())
                       )
    return    
def runmodel_edit_2d_call():
    runmodel_lenet2_edit (learning_rate = float(texto_ins22d.get()),
                         batch_size = int(texto_ins42d.get()),
                         num_epochs = int(texto_ins52d.get()),
                         PATIENCE = int(texto_ins62d.get()),
                         num_monte_carlo = int(texto_ins72d.get()),
                         inp1 = int(texto_ins82d.get()),
                         inp2 = int(texto_ins92d.get()),
                         inp3 = int(texto_ins02d.get()),
                         classes = int(texto_ins02d1.get())
                         )  
    return
    


#------------------ Save parameters Regression Lenet-AlexNet-Edit--------------
def run_regression_call():
    run_regression     (EXAMINE_AVERAGE= int(texto_ins72dR.get()),
                       THRESHOLD_EARLY_STOPING = float(texto_ins22dR.get()),
                       BATCH_SIZE = int(texto_ins42dR.get()),
                       NUMBER_EPOCHS = int(texto_ins52dR.get()),
                       PATIENCE_VALUE = int(texto_ins62dR.get()),
                       classes = int(texto_ins02d1R1.get()),
                       inputs = int(texto_ins02d1R.get())
                       )
    return
def run_regression_edit_call():
    run_regression_edit     (EXAMINE_AVERAGE= int(texto_ins72dR.get()),
                            THRESHOLD_EARLY_STOPING = float(texto_ins22dR.get()),
                            BATCH_SIZE = int(texto_ins42dR.get()),
                            NUMBER_EPOCHS = int(texto_ins52dR.get()),
                            PATIENCE_VALUE = int(texto_ins62dR.get()),
                            classes = int(texto_ins02d1R1.get()),
                            inputs = int(texto_ins02d1R.get())
                            )
    return
def run_regression_call1():
    run_regression_edit1     (EXAMINE_AVERAGE= int(texto_ins72dR.get()),
                            THRESHOLD_EARLY_STOPING = float(texto_ins22dR.get()),
                            BATCH_SIZE = int(texto_ins42dR.get()),
                            NUMBER_EPOCHS = int(texto_ins52dR.get()),
                            PATIENCE_VALUE = int(texto_ins62dR.get()),
                            classes = int(texto_ins02d1R1.get()),
                            inputs = int(texto_ins02d1R.get())
                            )
    return


#----------------- 1 dimension LeNet show results ----------------------------

def lenet_1d_all_plot_call():
    lenet_1d_all_plot(result = str(escolhegeneric.get()),
                                   classes = int(texto_ins22d.get())             
                                   )
    return
def lenet_1d_one_plot_call():
    lenet_1d_one_plot(result = str(escolhegeneric.get()),
                                classes = int(texto_ins22d.get())             
                                )
    return
def lenet_1d_print_call():
    lenet_1d_print(result = str(escolhegeneric.get()),
                                classes = int(texto_ins22d.get())             
                                )
    return


#---------------------- 1 dimension AlexNet show results  ----------------

def alexnet_1d_all_plot_call():
    alexnet_1d_all_plot(result = str(escolhegeneric.get()),
                                classes = int(texto_ins22d.get())             
                                )
    return
def alexnet_1d_one_plot_call():
    alexnet_1d_one_plot(result = str(escolhegeneric.get()),
                                classes = int(texto_ins22d.get())             
                                )
    return
def alexnet_1d_print_call():
    alexnet_1d_print(result = str(escolhegeneric.get()),
                                classes = int(texto_ins22d.get())             
                                )
    return


#----------------------- 1 dimension Edit show results ------------------------

def edit_1d_all_plot_call():
    edit_1d_all_plot(result = str(escolhegeneric.get()),
                                classes = int(texto_ins22d.get())             
                                )
    return
def edit_1d_one_plot_call():
    edit_1d_one_plot(result = str(escolhegeneric.get()),
                                classes = int(texto_ins22d.get())             
                                )
    return
def edit_1d_print_call():
    edit_1d_print(result = str(escolhegeneric.get()),
                                classes = int(texto_ins22d.get())             
                                )
    return


#---------------------2 dimensions LeNet show results ----------------------------------

def lenet_2d_all_plot_call():
    lenet_2d_all_plot(result = str(escolhegeneric.get()),
                                   classes = int(texto_ins22d.get())             
                                   )
    return
def lenet_2d_one_plot_call():
    lenet_2d_one_plot(result = str(escolhegeneric.get()),
                                classes = int(texto_ins22d.get())             
                                )
    return
def lenet_2d_print_call():
    lenet_2d_print(result = str(escolhegeneric.get()),
                                classes = int(texto_ins22d.get())             
                                )
    return


#---------------------- 2 dimensions AlexNet show results ----------

def alexnet_2d_all_plot_call():
    alexnet_2d_all_plot(result = str(escolhegeneric.get()),
                        classes = int(texto_ins22d.get())             
                                )
    return
def alexnet_2d_one_plot_call():
    alexnet_2d_one_plot(result = str(escolhegeneric.get()),
                                classes = int(texto_ins22d.get())             
                                )
    return
def alexnet_2d_print_call():
    alexnet_2d_print(result = str(escolhegeneric.get()),
                                classes = int(texto_ins22d.get())             
                                )
    return


#----------------------- 2 dimension Edit show results ------------------------

def edit_2d_all_plot_call():
    edit_2d_all_plot(result = str(escolhegeneric.get()),
                                classes = int(texto_ins22d.get())             
                                )
    return
def edit_2d_one_plot_call():
    edit_2d_one_plot(result = str(escolhegeneric.get()),
                                classes = int(texto_ins22d.get())             
                                )
    return
def edit_2d_print_call():
    edit_2d_print(result = str(escolhegeneric.get()),
                                classes = int(texto_ins22d.get())             
                                )
    return
               


#---- See Separator About -----------
def semComando():
    hide_all_frames()
    frame_menuSobre.grid()

#---- See Separator FAQ -----------
def verversao():
    hide_all_frames()
    frame_verversao.grid()

#-------- See Menu Start 1 dimension ------------------
def vermenuinformacao():
    hide_all_frames()
    frame_menu_informacao.grid()

#-------- See Menu Start 2 dimensions ------------------    
def vermenuinformacao1():
    hide_all_frames()
    frame_menu_informacao1.grid()

#-------- See Menu Start Regression ------------------        
def vermenuinformacaoR():
    hide_all_frames()
    frame_menu_informacaor.grid()


#------------ Aux Function to show menu insert parameters------------
def menuinserir():
    hide_all_frames()
    frame_menu_inserir.grid()  
def menuinserir2d():
    hide_all_frames()
    frame_menu_inserir2d.grid()   
def menuinserir1():
    hide_all_frames()
    frame_menu_inserir1.grid()  
def menuinserir12d():
    hide_all_frames()
    frame_menu_inserir12d.grid()
def menuinserir12dR():
    hide_all_frames()
    frame_menu_inserir12dR.grid()   
def menuinserir2():
    hide_all_frames()
    frame_menu_inserir2.grid()  
def menuinserir2_2d():
    hide_all_frames()
    frame_menu_inserir2_2d.grid()    
def menuinserir2_2dR():
    hide_all_frames()
    frame_menu_inserir2_2dR.grid()
def menuinserir22d():
    hide_all_frames()
    frame_menu_inserir22d.grid()    
def menuinserir3():
    hide_all_frames()
    frame_menu_inserir3.grid()   
def menuinserir32d():
    hide_all_frames()
    frame_menu_inserir3.grid()


# --------------------   Aux Functions to show Frames-------------------------
def genericresult():
    hide_all_frames()
    frame_menu_vergeneric.grid()
    genericresult_aux()
def runmodelgeneric1():
    hide_all_frames()
    frame_menu_runmodelgeneric1.grid()
    runmodelgeneric1_aux()
def genericresult1():
    hide_all_frames()
    frame_menu_vergeneric1.grid()
    genericresult1_aux()   
def genericresult2():
    hide_all_frames()
    frame_menu_vergeneric2.grid()
    genericresult2_aux()  
def alexnetresult():
    hide_all_frames()
    frame_menu_veralexnet.grid()
    alexnetresult_aux()    
def runmodelalexnet1():
    hide_all_frames()
    frame_menu_runmodelalexnet.grid()
    runmodelalexnet_aux()
def genericresult_edit():
    hide_all_frames()
    frame_menu_veralexnete2.grid()
    alexnetresult_aux_edit()    
def genericresult_edit1():
    hide_all_frames()
    frame_menu_veralexnete1.grid()
    alexnetresult_aux_edit1()
    
    

#-----------------  Clean Frames ------------------------------------------
def hide_all_frames():
    frame_menu_informacao.grid_forget()
    frame_menu_informacao1.grid_forget()
    frame_menu_inserir.grid_forget()
    frame_menu_inserir2d.grid_forget()
    frame_menu_inserir12dR.grid_forget()
    frame_menu_inserir1.grid_forget()
    frame_menu_inserir12d.grid_forget()
    frame_menu_inserir2.grid_forget()
    frame_menu_inserir2_2d.grid_forget()
    frame_menu_inserir2_2R.grid_forget()
    frame_menu_inserir2_2dR.grid_forget()
    frame_menu_inserir22d.grid_forget()
    frame_menu_inserir3.grid_forget()
    frame_menu_inserir32d.grid_forget()
    frame_menu_correredes.grid_forget()
    frame_menu_correredes1.grid_forget()
    frame_menu_escolhemodelo.grid_forget()
    frame_menu_vergeneric.grid_forget() 
    frame_menu_runmodelgeneric1.grid_forget() 
    frame_menuSobre.grid_forget()
    frame_verversao.grid_forget()
    frame_menu_verlenet.grid_forget()
    frame_menu_veralexnet.grid_forget()
    frame_menu_vergeneric1.grid_forget()
    frame_menu_vergeneric2.grid_forget()
    frame_menu_dense.grid_forget()
    frame_menu_pooling.grid_forget()
    frame_menu_convolucao.grid_forget()
    frame_menu_veralexnete2.grid_forget()
    frame_menu_veralexnete1.grid_forget()
    frame_menu_informacaor.grid_forget()
    
    

#-------------------------- Confirmation Insert parameters ------------------- 

#------------------------ 1 dimension parametes -------------------------------  
def Insere():
    SAMPLING_FREQ = int(texto_ins1.get())
    EXAMINE_AVERAGE= int(texto_ins2.get())
    THRESHOLD_EARLY_STOPING = float(texto_ins4.get())
    BATCH_SIZE = int(texto_ins5.get())
    NUMBER_EPOCHS = int(texto_ins6.get())
    PATIENCE_VALUE = int(texto_ins7.get())
    classes = int(texto_ins8.get())
    print("Parameters entered successfully")
    print("\nThe parameters that you insert are:")
    print("\nInput Frequency:", SAMPLING_FREQ) 
    print("Monte Carlo value", EXAMINE_AVERAGE) 
    print("Lerning rate", THRESHOLD_EARLY_STOPING) 
    print("Batch size:", BATCH_SIZE)  
    print("Maximum number of epoch:", NUMBER_EPOCHS)
    print("Patience value:", PATIENCE_VALUE) 
    print("Number of classes:", classes) 
    menuinserir2()

#------------------------ 2 dimensions parametes -----------------------------
def Insere2d():
    learning_rate = float(texto_ins22d.get())
    batch_size = int(texto_ins42d.get())
    num_epochs = int(texto_ins52d.get())
    PATIENCE = int(texto_ins62d.get())
    num_monte_carlo = int(texto_ins72d.get())
    inp1 = int(texto_ins82d.get()),
    inp2 = int(texto_ins92d.get()),
    inp3 = int(texto_ins02d.get()),
    classes = int(texto_ins02d1.get())
    print("Parameters entered successfully")
    print("Lerning rate", learning_rate) 
    print("Batch size:", batch_size) 
    print("Maximum number of epoch:", num_epochs)  
    print("Value for early stopping:", PATIENCE) 
    print("Monte Carlo value:", num_monte_carlo) 
    print("Classes:", classes)    
    print("Input Shape - width:", inp1)  
    print("Input Shape - height:", inp2) 
    print("Input Shape - channels:", inp3) 
    print("Number of classes:", classes)    
    menuinserir2_2d()

#------------------------ Regression parametes -----------------------------
def Insere2dR():
    learning_rate = float(texto_ins22dR.get())
    batch_size = int(texto_ins42dR.get())
    num_epochs = int(texto_ins52dR.get())
    PATIENCE = int(texto_ins62dR.get())
    num_monte_carlo = int(texto_ins72dR.get())
    inputs = int(texto_ins02d1R.get())
    classes = int(texto_ins02d1R1.get())
    print("Parameters entered successfully")
    print("Lerning rate", learning_rate) 
    print("Batch size:", batch_size) 
    print("Maximum number of epoch:", num_epochs)  
    print("Patience value:", PATIENCE) 
    print("Monte Carlo value:", num_monte_carlo) 
    print("Input Shape:", inputs) 
    print("Number of classes:", classes) 
    menuinserir2_2dR()
    
    
#-------------------   1d LeNet Result    -------------------------------------
def genericresult_aux():
    box_value3=StringVar()
    global escolhegeneric
    escolhegeneric = ttk.Combobox(frame_menu_vergeneric, textvariable=box_value3, state='readonly')
    escolhegeneric["values"] = ["Acc-LeNet","Auc-LeNet","Npv-LeNet","Ppv-LeNet","Sen-LeNet","Spe-LeNet"]
    escolhegeneric.current(0)
    label0 = Label(frame_menu_vergeneric, width=25, height=2,
                  text ="\nLeNet 1 Dimension \n", font=('verdana', 10, 'bold'), justify=CENTER)
    label01 = Label(frame_menu_vergeneric, width=25, height=2,
                  text ="\nClick the button to:\n", font=('verdana', 10, 'bold'), justify=CENTER)
    label02 = Label(frame_menu_vergeneric, width=25, height=3,
                  text ="\n\n-> See result of the select value:\n", font=('verdana', 11, 'bold'), justify=CENTER)
    label1 = Label(frame_menu_vergeneric, width=25, height=2,
                  text ="\n\n Select the Result \n", font=('verdana', 10), justify=CENTER)
    label0.grid(row=0, column=0)
    label01.grid(row=0, column=1)
    label02.grid(row=2, column=1)
    label1.grid(row=2, column=0)
    escolhegeneric.grid(row=3, column=0)
    Button(frame_menu_vergeneric,text="\nSee all the results in a plot", font=('verdana', 9, 'bold'), command=lenet_1d_all_plot_call).grid(row=1, column=1)
    Button(frame_menu_vergeneric,text="\nSee each result in a plot", font=('verdana', 9, 'bold'), command=lenet_1d_one_plot_call).grid(row=3, column=1)
    Button(frame_menu_vergeneric,text="\nPrint each result in Iptyton", font=('verdana', 9, 'bold'), command=lenet_1d_print_call).grid(row=4, column=1)
    labelclasse = Label(frame_menu_vergeneric, width=25, height=2,
                  text ="\n Number of classes: \n", font=('verdana', 10, 'bold'), justify=CENTER)
    labelclasse.grid(row=5, column=0)
    labelclasse1 = Label(frame_menu_vergeneric, width=25, height=1,
                  text ="  ", font=('verdana', 10, 'bold'), justify=CENTER)
    labelclasse1.grid(row=4, column=0)
    global texto_ins22d
    texto_ins22d = Entry(frame_menu_vergeneric, width=8, font="verdana 9", justify=CENTER)
    texto_ins22d.insert(END, '5')
    texto_ins22d.grid(row=6, column=0)


#-------------------   1d AlexNet Result    -----------------------------------
def alexnetresult_aux():
    box_value3=StringVar()
    global escolhegeneric
    escolhegeneric = ttk.Combobox(frame_menu_veralexnet, textvariable=box_value3, state='readonly')
    escolhegeneric["values"] = ["Acc-AlexNet","Auc-AlexNet","Npv-AlexNet","Ppv-AlexNet","Sen-AlexNet","Spe-AlexNet"]
    escolhegeneric.current(0)
    label0 = Label(frame_menu_veralexnet, width=25, height=2,
                  text ="\nAlexNet 1 Dimension \n", font=('verdana', 10, 'bold'), justify=CENTER)
    label01 = Label(frame_menu_veralexnet, width=25, height=2,
                  text ="\nClick the button to:\n", font=('verdana', 10, 'bold'), justify=CENTER)
    label02 = Label(frame_menu_veralexnet, width=25, height=3,
                  text ="\n\n-> See result of the select value:\n", font=('verdana', 11, 'bold'), justify=CENTER)
    label1 = Label(frame_menu_veralexnet, width=25, height=2,
                  text ="\n\n Select the Result \n", font=('verdana', 10), justify=CENTER)
    label0.grid(row=0, column=0)
    label01.grid(row=0, column=1)
    label02.grid(row=2, column=1)
    label1.grid(row=2, column=0)
    escolhegeneric.grid(row=3, column=0)
    Button(frame_menu_veralexnet,text="\nSee all the results in a plot", font=('verdana', 9, 'bold'), command=alexnet_1d_all_plot_call).grid(row=1, column=1)
    Button(frame_menu_veralexnet,text="\nSee each result in a plot", font=('verdana', 9, 'bold'), command=alexnet_1d_one_plot_call).grid(row=3, column=1)
    Button(frame_menu_veralexnet,text="\nPrint each result in Iptyton", font=('verdana', 9, 'bold'), command=alexnet_1d_print_call).grid(row=4, column=1)
    labelclasse = Label(frame_menu_veralexnet, width=25, height=2,
                  text ="\n Number of classes: \n", font=('verdana', 10, 'bold'), justify=CENTER)
    labelclasse.grid(row=5, column=0)
    labelclasse1 = Label(frame_menu_veralexnet, width=25, height=1,
                  text ="  ", font=('verdana', 10, 'bold'), justify=CENTER)
    labelclasse1.grid(row=4, column=0)
    global texto_ins22d
    texto_ins22d = Entry(frame_menu_veralexnet, width=8, font="verdana 9", justify=CENTER)
    texto_ins22d.insert(END, '5')
    texto_ins22d.grid(row=6, column=0)
    
    
#------------------------------  1d  Edit Model      --------------------------

def alexnetresult_aux_edit1():
    box_value3=StringVar()
    global escolhegeneric
    escolhegeneric = ttk.Combobox(frame_menu_veralexnete1, textvariable=box_value3, state='readonly')
    escolhegeneric["values"] = ["Acc-Edit1d","Auc-Edit1d","Npv-Edit1d","Ppv-Edit1d","Sen-Edit1d","Spe-Edit1d"]
    escolhegeneric.current(0)
    label0 = Label(frame_menu_veralexnete1, width=25, height=2,
                  text ="\nEdit 1 Dimension \n", font=('verdana', 10, 'bold'), justify=CENTER)
    label01 = Label(frame_menu_veralexnete1, width=25, height=2,
                  text ="\nClick the button to:\n", font=('verdana', 10, 'bold'), justify=CENTER)
    label02 = Label(frame_menu_veralexnete1, width=25, height=3,
                  text ="\n\n-> See result of the select value:\n", font=('verdana', 11, 'bold'), justify=CENTER)
    label1 = Label(frame_menu_veralexnete2, width=25, height=2,
                  text ="\n\n Select the Result \n", font=('verdana', 10), justify=CENTER)
    label0.grid(row=0, column=0)
    label01.grid(row=0, column=1)
    label02.grid(row=2, column=1)
    label1.grid(row=2, column=0)
    escolhegeneric.grid(row=3, column=0)
    Button(frame_menu_veralexnete1,text="\nSee all the results in a plot", font=('verdana', 9, 'bold'), command=edit_1d_all_plot_call).grid(row=1, column=1)
    Button(frame_menu_veralexnete1,text="\nSee each result in a plot", font=('verdana', 9, 'bold'), command=edit_1d_one_plot_call).grid(row=3, column=1)
    Button(frame_menu_veralexnete1,text="\nPrint each result in Iptyton", font=('verdana', 9, 'bold'), command=edit_1d_print_call).grid(row=4, column=1)
    labelclasse = Label(frame_menu_veralexnete1, width=25, height=2,
                  text ="\n Number of classes: \n", font=('verdana', 10, 'bold'), justify=CENTER)
    labelclasse.grid(row=5, column=0)
    labelclasse1 = Label(frame_menu_veralexnete1, width=25, height=1,
                  text ="  ", font=('verdana', 10, 'bold'), justify=CENTER)
    labelclasse1.grid(row=4, column=0)
    global texto_ins22d
    texto_ins22d = Entry(frame_menu_veralexnete1, width=8, font="verdana 9", justify=CENTER)
    texto_ins22d.insert(END, '5')
    texto_ins22d.grid(row=6, column=0)
    

#-------------------   2d LeNet Result ---------------------------------------

def genericresult1_aux():
    box_value3=StringVar()
    global escolhegeneric
    escolhegeneric = ttk.Combobox(frame_menu_vergeneric1, textvariable=box_value3, state='readonly')
    escolhegeneric["values"] = ["Acc-LeNet","Auc-LeNet","Npv-LeNet","Ppv-LeNet","Sen-LeNet","Spe-LeNet"]
    escolhegeneric.current(0)
    label0 = Label(frame_menu_vergeneric1, width=25, height=2,
                  text ="\nLeNet 2 Dimensions \n", font=('verdana', 10, 'bold'), justify=CENTER)
    label01 = Label(frame_menu_vergeneric1, width=25, height=2,
                  text ="\nClick the button to:\n", font=('verdana', 10, 'bold'), justify=CENTER)
    label02 = Label(frame_menu_vergeneric1, width=25, height=3,
                  text ="\n\n-> See result of the select value:\n", font=('verdana', 11, 'bold'), justify=CENTER)
    label1 = Label(frame_menu_vergeneric1, width=25, height=2,
                  text ="\n\n Select the Result \n", font=('verdana', 10), justify=CENTER)
    label0.grid(row=0, column=0)
    label01.grid(row=0, column=1)
    label02.grid(row=2, column=1)
    label1.grid(row=2, column=0)
    escolhegeneric.grid(row=3, column=0)
    Button(frame_menu_vergeneric1,text="\nSee all the results in a plot", font=('verdana', 9, 'bold'), command=lenet_2d_all_plot_call).grid(row=1, column=1)
    Button(frame_menu_vergeneric1,text="\nSee each result in a plot", font=('verdana', 9, 'bold'), command=lenet_2d_one_plot_call).grid(row=3, column=1)
    Button(frame_menu_vergeneric1,text="\nPrint each result in Iptyton", font=('verdana', 9, 'bold'), command=lenet_2d_print_call).grid(row=4, column=1)
    labelclasse = Label(frame_menu_vergeneric1, width=25, height=2,
                  text ="\n Number of classes: \n", font=('verdana', 10, 'bold'), justify=CENTER)
    labelclasse.grid(row=5, column=0)
    labelclasse1 = Label(frame_menu_vergeneric1, width=25, height=1,
                  text ="  ", font=('verdana', 10, 'bold'), justify=CENTER)
    labelclasse1.grid(row=4, column=0)
    global texto_ins22d
    texto_ins22d = Entry(frame_menu_vergeneric1, width=8, font="verdana 9", justify=CENTER)
    texto_ins22d.insert(END, '10')
    texto_ins22d.grid(row=6, column=0)


    
#-------------------   2d AlexNet Result --------------------------------------

def genericresult2_aux():
    box_value3=StringVar()
    texto = Entry(frame_menu_vergeneric2, width=8, font="verdana 9", justify=CENTER)
    texto.insert(END,"Acc-AlexNet")
    texto.grid(row=3,column=0)
    global escolhegeneric
    escolhegeneric = ttk.Combobox(frame_menu_vergeneric2, textvariable=box_value3, state='readonly')
    escolhegeneric["values"] = ["Acc-AlexNet","Auc-AlexNet","Npv-AlexNet","Ppv-AlexNet","Sen-AlexNet","Spe-AlexNet"]
    escolhegeneric.current(0)
    label0 = Label(frame_menu_vergeneric2, width=25, height=2,
                  text ="\nAlexNet 2 Dimensions \n", font=('verdana', 10, 'bold'), justify=CENTER)
    label01 = Label(frame_menu_vergeneric2, width=25, height=2,
                  text ="\nClick the button to:\n", font=('verdana', 10, 'bold'), justify=CENTER)
    label02 = Label(frame_menu_vergeneric2, width=25, height=3,
                  text ="\n\n-> See result of the select value:\n", font=('verdana', 11, 'bold'), justify=CENTER)
    label1 = Label(frame_menu_vergeneric2, width=25, height=2,
                  text ="\n\n Select the Result \n", font=('verdana', 10), justify=CENTER)
    label0.grid(row=0, column=0)
    label01.grid(row=0, column=1)
    label02.grid(row=2, column=1)
    label1.grid(row=2, column=0)
    escolhegeneric.grid(row=3, column=0)
    Button(frame_menu_vergeneric2,text="\nSee all the results in a plot", font=('verdana', 9, 'bold'), command=alexnet_2d_all_plot_call).grid(row=1, column=1)
    Button(frame_menu_vergeneric2,text="\nSee each result in a plot", font=('verdana', 9, 'bold'), command=alexnet_2d_one_plot_call).grid(row=3, column=1)
    Button(frame_menu_vergeneric2,text="\nPrint each result in Iptyton", font=('verdana', 9, 'bold'), command=alexnet_2d_print_call).grid(row=4, column=1)
    labelclasse = Label(frame_menu_vergeneric2, width=25, height=2,
                  text ="\n Number of classes: \n", font=('verdana', 10, 'bold'), justify=CENTER)
    labelclasse.grid(row=5, column=0)
    labelclasse1 = Label(frame_menu_vergeneric2, width=25, height=1,
                  text ="  ", font=('verdana', 10, 'bold'), justify=CENTER)
    labelclasse1.grid(row=4, column=0)
    global texto_ins22d
    texto_ins22d = Entry(frame_menu_vergeneric2, width=8, font="verdana 9", justify=CENTER)
    texto_ins22d.insert(END, '10')
    texto_ins22d.grid(row=6, column=0)
    
    
#----------------------------- 1d  Edit Model  --------------------------------

def alexnetresult_aux_edit():
    box_value3=StringVar()
    global escolhegeneric
    escolhegeneric = ttk.Combobox(frame_menu_veralexnete2, textvariable=box_value3, state='readonly')
    escolhegeneric["values"] = ["Acc-Edit2d","Auc-Edit2d","Npv-Edit2d","Ppv-Edit2d","Sen-Edit2d","Spe-Edit2d"]
    escolhegeneric.current(0)
    label0 = Label(frame_menu_veralexnete2, width=25, height=2,
                  text ="\nEdit 2 Dimensions \n", font=('verdana', 10, 'bold'), justify=CENTER)
    label01 = Label(frame_menu_veralexnete2, width=25, height=2,
                  text ="\nClick the button to:\n", font=('verdana', 10, 'bold'), justify=CENTER)
    label02 = Label(frame_menu_veralexnete2, width=25, height=3,
                  text ="\n\n-> See result of the select value:\n", font=('verdana', 11, 'bold'), justify=CENTER)
    label1 = Label(frame_menu_veralexnete2, width=25, height=2,
                  text ="\n\n Select the Result \n", font=('verdana', 10), justify=CENTER)
    label0.grid(row=0, column=0)
    label01.grid(row=0, column=1)
    label02.grid(row=2, column=1)
    label1.grid(row=2, column=0)
    escolhegeneric.grid(row=3, column=0)
    Button(frame_menu_veralexnete2,text="\nSee all the results in a plot", font=('verdana', 9, 'bold'), command=edit_2d_all_plot_call).grid(row=1, column=1)
    Button(frame_menu_veralexnete2,text="\nSee each result in a plot", font=('verdana', 9, 'bold'), command=edit_2d_one_plot_call).grid(row=3, column=1)
    Button(frame_menu_veralexnete2,text="\nPrint each result in Iptyton", font=('verdana', 9, 'bold'), command=edit_2d_print_call).grid(row=4, column=1)
    labelclasse = Label(frame_menu_veralexnete2, width=25, height=2,
                  text ="\n Number of classes: \n", font=('verdana', 10, 'bold'), justify=CENTER)
    labelclasse.grid(row=5, column=0)
    labelclasse1 = Label(frame_menu_veralexnete2, width=25, height=1,
                  text ="  ", font=('verdana', 10, 'bold'), justify=CENTER)
    labelclasse1.grid(row=4, column=0)
    global texto_ins22d
    texto_ins22d = Entry(frame_menu_veralexnete2, width=8, font="verdana 9", justify=CENTER)
    texto_ins22d.insert(END, '10')
    texto_ins22d.grid(row=6, column=0)



#-------------- Imprime na consola para confirmar
    
menu_inicial = Tk()
menu_inicial.title("IG for development probabilistic CNNs")
 
#---------------------------GUI ------------------------

#dimensóes da janela
largura = 700
altura = 500

#resolução do nosso sistema
largura_screen = menu_inicial.winfo_screenwidth()
altura_screen = menu_inicial.winfo_screenheight()

#print(largura_screen, altura_screen)

#Position of the window
posx = largura_screen/2 - largura/2
posy = altura_screen/2 - altura/2

#Tamanho do menu - definir a geometry
menu_inicial.geometry("%dx%d+%d+%d" % (largura, altura, posx, posy))

#------------------- Create Frame ---------------------------------------------

frame_menuSobre = Frame(menu_inicial)
frame_verversao = Frame(menu_inicial)
frame_menu_informacao = Frame(menu_inicial)
frame_menu_informacao1 = Frame(menu_inicial)
frame_menu_inserir =  Frame(menu_inicial)
frame_menu_inserir1 =  Frame(menu_inicial)
frame_menu_inserir2 =  Frame(menu_inicial)
frame_menu_inserir2_2d =  Frame(menu_inicial)
frame_menu_inserir2_2dR =  Frame(menu_inicial)
frame_menu_inserir3 =  Frame(menu_inicial)
frame_menu_inserir2d =  Frame(menu_inicial)
frame_menu_inserir12dR =  Frame(menu_inicial)
frame_menu_inserir2_2R =  Frame(menu_inicial)
frame_menu_inserir12d =  Frame(menu_inicial)
frame_menu_inserir22d =  Frame(menu_inicial)
frame_menu_inserir32d =  Frame(menu_inicial)
frame_menu_correredes = Frame(menu_inicial)
frame_menu_correredes1 = Frame(menu_inicial)
frame_menu_escolhemodelo = Frame(menu_inicial)
frame_menu_vergeneric = Frame(menu_inicial)
frame_menu_runmodelgeneric1 = Frame(menu_inicial)
frame_menu_verlenet = Frame(menu_inicial)
frame_menu_veralexnet = Frame(menu_inicial)
frame_menu_alexnet_runmodelgeneric1 = Frame(menu_inicial)
frame_menu_vergeneric1 = Frame(menu_inicial)
frame_menu_vergeneric2 = Frame(menu_inicial)
frame_menu_dense = Frame(menu_inicial)
frame_menu_convolucao  = Frame(menu_inicial)
frame_menu_pooling  = Frame(menu_inicial)
frame_menu_veralexnete2 = Frame(menu_inicial)
frame_menu_veralexnete1 = Frame(menu_inicial)
frame_menu_informacaor = Frame(menu_inicial)


#---------------------------- Gui Barra de Menus -------------------------------

barraDeMenus = Menu(menu_inicial)
menuinformacao = Menu(barraDeMenus,tearoff=0)
menuinformacao.add_command(label="See how to Start 1 Dimensions Datasets",command=vermenuinformacao)
menuinformacao.add_command(label="See how to Start 2 Dimensions Datasets",command=vermenuinformacao1)
menuinformacao.add_command(label="See how to Regression",command=vermenuinformacaoR)
barraDeMenus.add_cascade(label="Start",menu=menuinformacao)


#------------------------       Separador Insere          -------------------
menuinsere = Menu(barraDeMenus,tearoff=0)
menuinsere.add_command(label="1 Dimension parameters",command=menuinserir1)
menuinsere.add_command(label="2 Dimensions parameters",command=menuinserir12d)
menuinsere.add_command(label="Regression",command=menuinserir12dR)
menuinsere.add_separator()
barraDeMenus.add_cascade(label="Insert | Run",menu=menuinsere)


#------------------------       Separador Resultados      -------------------
menuresultados = Menu(barraDeMenus,tearoff=0)
menuresultados.add_command(label="1 Dimnension - LeNet",command=genericresult)
menuresultados.add_command(label="1 Dimension - AlexNet",command=alexnetresult)
menuresultados.add_command(label="2 Dimensions - LeNet",command=genericresult1)
menuresultados.add_command(label="2 Dimension - AlexNet",command=genericresult2)
menuresultados.add_command(label="Edit 1 Dimension",command=genericresult_edit1)
menuresultados.add_command(label="Edit 2 Dimensions",command=genericresult_edit)
menuresultados.add_separator()
barraDeMenus.add_cascade(label="Results",menu=menuresultados)


#------------------------       Separador FAQ     -------------------

menuresultados = Menu(barraDeMenus,tearoff=0)
menuresultados.add_command(label="Software | Versions Compatible",command=verversao)
menuresultados.add_separator()
barraDeMenus.add_cascade(label="FAQ",menu=menuresultados)


#------------------------       Separador About      -------------------

menuSobre = Menu(barraDeMenus,tearoff=0)
menuSobre.add_command(label="Show Message",command=semComando)
menuSobre.add_separator()
barraDeMenus.add_cascade(label="About",menu=menuSobre)
menu_inicial.config(menu=barraDeMenus)



#---------------------------- Menu FAQ        ------------- 

label_mensagem_versao = Label(frame_verversao,
                  text ="\n\n   Programs and list of packages", font="verdana 10 bold")
label_mensagem1_versao = Label(frame_verversao,
                  text ="\n Versions used:", font="verdana 10")
label_mensagem2_versao = Label(frame_verversao,
                  text ="\n CudNN = 8.2 | CUDA = 11.4 | TensorFlow = 2.6.0 | Phyton = 3.8.8", font="verdana 10")
label_mensagem3_versao = Label(frame_verversao,
                  text ="\n      tensorflow-probability = 0.14.1 | Tkinter = 8.6 | scikit-learn = 0.24.1 ", font="verdana 10")
label_mensagem4_versao = Label(frame_verversao,
                  text ="\n h5py = 3.1.0 | numpy = 1.19.5 | pandas = 1.2.4 | pycm==3.2 ", font="verdana 10")
label_mensagem5_versao = Label(frame_verversao,
                  text ="\n sympy = 1.8 | keras = 2.6.0 | matplotlib = 3.3.4 | scipy = 1.4.1| ", font="verdana 10")
label_mensagem6_versao = Label(frame_verversao,
                  text ="\n\n If you want to see some videos press the button:\n", font="verdana 10")
new = 1
url = "https://www.youtube.com/channel/UC9bZjefkicHKC6VJPGLLZJA/videos"
def openweb():
    webbrowser.open(url,new=new)
button = Button(frame_verversao, text = "See Videos",command=openweb, font="verdana 9 bold")
label_mensagem_versao.grid(row=0, column=0)
label_mensagem1_versao.grid(row=1, column=0)
label_mensagem2_versao.grid(row=2, column=0)
label_mensagem3_versao.grid(row=3, column=0)
label_mensagem4_versao.grid(row=4, column=0)
label_mensagem5_versao.grid(row=5, column=0)
label_mensagem6_versao.grid(row=6, column=0)
button.grid(row=7, column=0)


#---------------------------- Menu inicial IRUC DATASET ------------- 
label_mensagem_sobre = Label(frame_menuSobre,
                  text ="\n\n          Graphical Interface for development of CNN Probabilistics", font="verdana 10 bold")
label_mensagem1_sobre = Label(frame_menuSobre,
                  text ="\n Program developed by:", font="verdana 10")
label_mensagem2_sobre = Label(frame_menuSobre,
                  text ="\n     Aníbal João Lopes Chaves", font="verdana 10 bold")
label_mensagem21_sobre = Label(frame_menuSobre,
                  text ="\n September of 2021 - June of 2022", font="verdana 10")


label_mensagem_sobre.grid(row=0, column=0)
label_mensagem1_sobre.grid(row=1, column=0)
label_mensagem2_sobre.grid(row=2, column=0)
label_mensagem21_sobre.grid(row=3, column=0)




#---------------------------- Inicial Menu 1 dimension --------------
    

label_mensagem = Label(frame_menu_informacao,
                  text ="\n 1 - Preparing your dataset.", font="verdana 10 bold")
label_mensagem1 = Label(frame_menu_informacao,
                  text ="\n To use this application your data must consist of 4 csv files: ", font="verdana 9")
label_mensagem2 = Label(frame_menu_informacao,
                  text ="     x_test_data.csv | x_train_data.csv | y_test_label.csv | y_train_label.csv", font="verdana 9 bold")
label_mensagem21 = Label(frame_menu_informacao,
                  text ="Note: Put the CSV files in the same folder of the Program", font="verdana 9")
label_mensagem3 = Label(frame_menu_informacao,
                  text ="\n\n 2 - Insert | Run.", font="verdana 10 bold")
label_mensagem4 = Label(frame_menu_informacao,
                  text ="\n In the Program separator Insert | Run,", font="verdana 9")
label_mensagem5 = Label(frame_menu_informacao,
                  text ="\n Insert the parametres and after run", font="verdana 9 bold")
label_mensagem6 = Label(frame_menu_informacao,
                  text ="\n\n 3 - Results, choose the results for each Arquitecture", font="verdana 10 bold")
label_mensagem7 = Label(frame_menu_informacao,
                  text ="\n Show PLot - See  each/all the results in a plot ", font="verdana 9")
label_mensagem8 = Label(frame_menu_informacao,
                  text =" Print each result in Iphyton", font="verdana 9")


label_mensagem.grid(row=0, column=0)
label_mensagem1.grid(row=1, column=0)
label_mensagem2.grid(row=2, column=0)
label_mensagem21.grid(row=3, column=0)
label_mensagem3.grid(row=4, column=0)
label_mensagem4.grid(row=5, column=0)
label_mensagem5.grid(row=6, column=0)
label_mensagem6.grid(row=7, column=0)
label_mensagem7.grid(row=8, column=0)
label_mensagem8.grid(row=9, column=0)


#---------------------------- Start Menu 1 dimension Regression --------------
    
label_mensagem3_R = Label(frame_menu_informacaor,
                  text ="\n\n              1 - Select | Run Parameters and Arquitectures.", font="verdana 10 bold")
label_mensagem4_R = Label(frame_menu_informacaor,
                  text ="\n                In the Program separator Insert | Run (Regression)", font="verdana 9")
label_mensagem5_R = Label(frame_menu_informacaor,
                  text ="\n                Insert the parametres and after edit|run", font="verdana 9 bold")
label_mensagem9_R = Label(frame_menu_informacaor,
                  text ="\n\n              Go to Folder Results to see other results", font="verdana 10 bold")



label_mensagem3_R.grid(row=4, column=0)
label_mensagem4_R.grid(row=5, column=0)
label_mensagem5_R.grid(row=6, column=0)
label_mensagem9_R.grid(row=10, column=0)

#---------------------------- Star Menu two dimensions --------------
    
label_mensagem3_MNIST = Label(frame_menu_informacao1,
                  text ="\n\n              1 - Select | Run Parameters and Arquitectures.", font="verdana 10 bold")
label_mensagem4_MNIST = Label(frame_menu_informacao1,
                  text ="\n                In the Program separator Insert | Run (2 dimensions)", font="verdana 9")
label_mensagem5_MNIST = Label(frame_menu_informacao1,
                  text ="\n                Insert the parametres and after edit|run", font="verdana 9 bold")
label_mensagem6_MNIST = Label(frame_menu_informacao1,
                  text ="\n\n              2 - Results, choose the results for each Arquitecture", font="verdana 10 bold")
label_mensagem7_MNIST = Label(frame_menu_informacao1,
                  text ="\n                Show PLot - See  each/all the results in a plot ", font="verdana 9")
label_mensagem8_MNIST = Label(frame_menu_informacao1,
                  text ="                  Print each result in Iphyton", font="verdana 9")
label_mensagem9_MNIST = Label(frame_menu_informacao1,
                  text ="\n\n              Go to Folder Results to see other results", font="verdana 10 bold")



label_mensagem3_MNIST.grid(row=4, column=0)
label_mensagem4_MNIST.grid(row=5, column=0)
label_mensagem5_MNIST.grid(row=6, column=0)
label_mensagem6_MNIST.grid(row=7, column=0)
label_mensagem7_MNIST.grid(row=8, column=0)
label_mensagem8_MNIST.grid(row=9, column=0)
label_mensagem9_MNIST.grid(row=10, column=0)

0
#----------------------------------- Widgets message ------------------------------------------------------------

label_ins1_insere3 = Label(frame_menu_inserir3, width=33, height=2,
                      text ="Check the progress in the Console.", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins_insere3 = Label(frame_menu_inserir3, width=33, height=3,
                      text ="This process can take several hours.", font=('verdana', 10), justify=CENTER)
label_ins1_insere3.grid(row=0, column=0)
label_ins_insere3.grid(row=1, column=0)


#----------------------------------- Widgets Select one dimension AlexNet/ Lenet or Train --------------------

label_ins1_insere2 = Label(frame_menu_inserir2, width=33, height=2,
                      text ="Check the progress in the Console.", font=('verdana', 11, 'bold'), justify=CENTER)
label_ins_insere2 = Label(frame_menu_inserir2, width=33, height=3,
                      text ="Press the button to choose the Model:", font=('verdana', 11), justify=CENTER)
label_ins1_insere2.grid(row=0, column=0)
label_ins_insere2.grid(row=1, column=0)
cmd_escolhemodelo = Button(frame_menu_inserir2, text="Run LeNet-5",
                                 font=('verdana', 9, 'bold'),command=runmodel_lenet_1d_call)
cmd_escolhemodelo.grid(row=2,column=0)
cmd_editamodelo = Button(frame_menu_inserir2, text="Run Alexnet",
                                 font=('verdana', 9, 'bold'),command=alexnet_resultado_generic1_call)
cmd_editamodelo.grid(row=2,column=1)
label_ins_insere1_edit1 = Label(frame_menu_inserir2, width=33, height=3,
                      text ="\n\n Press the button to train the Model:\n", font=('verdana', 11), justify=CENTER)
label_ins_insere1_edit1.grid(row=3, column=0)

cmd_escolhemodelo1_edit1 = Button(frame_menu_inserir2, text="Train the Model",
                                 font=('verdana', 9, 'bold'),command=runmodel_edit_1d_call)
cmd_escolhemodelo1_edit1.grid(row=4,column=1)
label_ins_insere1_editb1 = Label(frame_menu_inserir2, width=33, height=3,
                      text ="\n\n Press the button to change parameters:\n", font=('verdana', 11), justify=CENTER)
label_ins_insere1_editb1.grid(row=5, column=0)
cmd_escolhemodelo1b_editb1 = Button(frame_menu_inserir2, text="Back",
                                 font=('verdana', 9, 'bold'),command=menuinserir1)
cmd_escolhemodelo1b_editb1.grid(row=6,column=1)



#----------------------------------- Widgets Select Regression AlexNet/ Lenet or Train --------------------

label_ins1_insere2_2dR = Label(frame_menu_inserir2_2dR, width=33, height=2,
                      text ="Check the progress in the Console.", font=('verdana', 11, 'bold'), justify=CENTER)
label_ins_insere2_2dR = Label(frame_menu_inserir2_2dR, width=33, height=3,
                      text ="Press the button to choose the Model:", font=('verdana', 11), justify=CENTER)
label_ins1_insere2_2dR.grid(row=0, column=0)
label_ins_insere2_2dR.grid(row=1, column=0)
cmd_escolhemodelo_2dR = Button(frame_menu_inserir2_2dR, text="Run LeNet-5",
                                 font=('verdana', 9, 'bold'),command=run_regression_call)
cmd_escolhemodelo_2dR.grid(row=2,column=0)
cmd_escolhemodelo_2dR1 = Button(frame_menu_inserir2_2dR, text="Run AlexNet",
                                 font=('verdana', 9, 'bold'),command=run_regression_call1)
cmd_escolhemodelo_2dR1.grid(row=2,column=1)
label_ins_insere1_editR = Label(frame_menu_inserir2_2dR, width=33, height=3,
                      text ="\n\n Press the button to train the Model:\n", font=('verdana', 11), justify=CENTER)
label_ins_insere1_editR.grid(row=3, column=0)
cmd_escolhemodelo1_editR = Button(frame_menu_inserir2_2dR, text="Train Modelt",
                                 font=('verdana', 9, 'bold'),command=run_regression_edit_call)
cmd_escolhemodelo1_editR.grid(row=4,column=1)
label_ins_insere1_editbR = Label(frame_menu_inserir2_2dR, width=33, height=3,
                      text ="\n\n Press the button to change parameters:\n", font=('verdana', 11), justify=CENTER)
label_ins_insere1_editbR.grid(row=5, column=0)
cmd_escolhemodelo1b_editbR = Button(frame_menu_inserir2_2dR, text="Back",
                                 font=('verdana', 9, 'bold'),command=menuinserir12dR)
cmd_escolhemodelo1b_editbR.grid(row=6,column=1)

#----------------------------------- Widgets Select model 2d AlexNet/ Lenet or Train --------------------

label_ins1_insere2_2d = Label(frame_menu_inserir2_2d, width=33, height=2,
                      text ="Check the progress in the Console.", font=('verdana', 11, 'bold'), justify=CENTER)
label_ins_insere2_2d = Label(frame_menu_inserir2_2d, width=33, height=3,
                      text ="Press the button to choose the Model:", font=('verdana', 11), justify=CENTER)
label_ins1_insere2_2d.grid(row=0, column=0)
label_ins_insere2_2d.grid(row=1, column=0)
cmd_escolhemodelo_2d = Button(frame_menu_inserir2_2d, text="Run LeNet-5",
                                 font=('verdana', 9, 'bold'),command=runmodel_lenet_2d_call)
cmd_escolhemodelo_2d.grid(row=2,column=0)
cmd_editamodelo_2d = Button(frame_menu_inserir2_2d, text="Run Alexnet",
                                 font=('verdana', 9, 'bold'),command=runmodel_alexlenet_2d_call)
cmd_editamodelo_2d.grid(row=2,column=1)
label_ins_insere1_edit = Label(frame_menu_inserir2_2d, width=33, height=3,
                      text ="\n\n Press the button to train the Model:\n", font=('verdana', 11), justify=CENTER)
label_ins_insere1_edit.grid(row=3, column=0)
cmd_escolhemodelo1_edit = Button(frame_menu_inserir2_2d, text="Train the Model",
                                 font=('verdana', 9, 'bold'),command=runmodel_edit_2d_call)
cmd_escolhemodelo1_edit.grid(row=4,column=1)
label_ins_insere1_editb = Label(frame_menu_inserir2_2d, width=33, height=3,
                      text ="\n\n Press the button to change parameters:\n", font=('verdana', 11), justify=CENTER)
label_ins_insere1_editb.grid(row=5, column=0)
cmd_escolhemodelo1b_editb = Button(frame_menu_inserir2_2d, text="Back",
                                 font=('verdana', 9, 'bold'),command=menuinserir12d)
cmd_escolhemodelo1b_editb.grid(row=6,column=1)

#------------------------------------------------Widgets Insert 2 dimensions------------------------------------------------------

label_ins = Label(frame_menu_inserir1, width=33, height=2,
                  text ="Insert Parameters for 1 dimension", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins0 = Label(frame_menu_inserir1, width=8, height=2,
                  text ="Values:", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins01 = Label(frame_menu_inserir1, width=15, height=1,
                  text ="Recomended:", font=('verdana', 8, 'bold'), justify=CENTER)
label_ins1 = Label(frame_menu_inserir1, width=33, height=1,
                  text ="Input frequency: ", font="verdana 10")
label_ins1_1 = Label(frame_menu_inserir1, width=10, height=1,
                  text ="200", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins2 = Label(frame_menu_inserir1, width=33, height=1,
                  text ="Monte Carlo:", font="verdana 10")
label_ins2_1 = Label(frame_menu_inserir1, width=10, height=1,
                  text ="20", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins4 = Label(frame_menu_inserir1, width=33, height=1,
                  text ="Lerning Rate:", font="verdana 10")
label_ins4_1 = Label(frame_menu_inserir1, width=10, height=1,
                  text ="0.005", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins5 = Label(frame_menu_inserir1, width=33, height=1,
                  text ="Batch Size:", font="verdana 10")
label_ins5_1 = Label(frame_menu_inserir1, width=10, height=1,
                  text ="32..256", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins6 = Label(frame_menu_inserir1, width=33, height=1,
                  text ="Maximum number of epochs:", font="verdana 10")
label_ins6_1 = Label(frame_menu_inserir1, width=10, height=1,
                  text ="400", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins7 = Label(frame_menu_inserir1, width=33, height=1,
                  text ="Patience value:", font="verdana 10")
label_ins7_1 = Label(frame_menu_inserir1, width=10, height=1,
                  text ="40", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins7r1 = Label(frame_menu_inserir1, width=33, height=1,
                  text ="Number of classes:", font="verdana 10")
label_ins7_1r = Label(frame_menu_inserir1, width=10, height=1,
                  text ="5", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins11r = Label(frame_menu_inserir1, width=33, height=1,
                  text ="                ", font="verdana 10")


texto_ins1 = Entry(frame_menu_inserir1, width=8, font="verdana 9", justify=CENTER)
texto_ins1.insert(END, '200') # Preencher por defeito 200
texto_ins2 = Entry(frame_menu_inserir1, width=8, font="verdana 9", justify=CENTER)
texto_ins2.insert(END, '20')
texto_ins4 = Entry(frame_menu_inserir1, width=8, font="verdana 9", justify=CENTER)
texto_ins4.insert(END, '0.005')
texto_ins5 = Entry(frame_menu_inserir1, width=8, font="verdana 9", justify=CENTER)
texto_ins5.insert(END, '32')
texto_ins6 = Entry(frame_menu_inserir1, width=8, font="verdana 9", justify=CENTER)
texto_ins6.insert(END, '400')
texto_ins7 = Entry(frame_menu_inserir1, width=8, font="verdana 9", justify=CENTER)
texto_ins7.insert(END, '40')
texto_ins8 = Entry(frame_menu_inserir1, width=8, font="verdana 9", justify=CENTER)
texto_ins8.insert(END, '5')

label_ins.grid(row=0, column=0)
label_ins0.grid(row=0, column=1)
label_ins01.grid(row=0, column=2)
label_ins1.grid(row=1, column=0)
label_ins1_1.grid(row=1, column=2)
label_ins2.grid(row=2, column=0)
label_ins2_1.grid(row=2, column=2)
label_ins4.grid(row=4, column=0)
label_ins4_1.grid(row=4, column=2)
label_ins5.grid(row=5, column=0)
label_ins5_1.grid(row=5, column=2)
label_ins6.grid(row=6, column=0)
label_ins6_1.grid(row=6, column=2)
label_ins7.grid(row=7, column=0)
label_ins7r1.grid(row=8, column=0)
label_ins7_1.grid(row=7, column=2)
label_ins7_1r.grid(row=8, column=2)
label_ins11r.grid(row=9, column=0)

texto_ins1.grid(row=1,column=1)
texto_ins2.grid(row=2,column=1)
texto_ins4.grid(row=4,column=1)
texto_ins5.grid(row=5,column=1)
texto_ins6.grid(row=6,column=1)
texto_ins7.grid(row=7,column=1)
texto_ins8.grid(row=8,column=1)

cmd_verificavalores = Button(frame_menu_inserir1, text="Insert/Save",
                             font=('verdana', 9, 'bold'),command=Insere)
cmd_verificavalores.grid(row=11,column=1)


#----------------------------------- Widget Insert  parameters 2 dimensions------------------------------------------------------

label_ins2d = Label(frame_menu_inserir12d, width=33, height=2,
                  text ="Insert Parameters for 2 dimensions", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins02d = Label(frame_menu_inserir12d, width=8, height=2,
                  text ="Values:", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins012d = Label(frame_menu_inserir12d, width=15, height=1,
                  text ="Recomended:", font=('verdana', 8, 'bold'), justify=CENTER)
label_ins22d = Label(frame_menu_inserir12d, width=33, height=1,
                  text ="Learning rate:", font="verdana 10")
label_ins2_12d = Label(frame_menu_inserir12d, width=10, height=1,
                  text ="0.001", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins42d = Label(frame_menu_inserir12d, width=33, height=1,
                  text ="Batch Size:", font="verdana 10")
label_ins4_12d = Label(frame_menu_inserir12d, width=10, height=1,
                  text ="16...128", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins52d = Label(frame_menu_inserir12d, width=33, height=1,
                  text ="Maximun number of epoch:", font="verdana 10")
label_ins5_12d = Label(frame_menu_inserir12d, width=10, height=1,
                  text ="300", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins62d = Label(frame_menu_inserir12d, width=33, height=1,
                  text ="Patience value:", font="verdana 10")
label_ins6_12d = Label(frame_menu_inserir12d, width=10, height=1,
                  text ="10", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins72d = Label(frame_menu_inserir12d, width=33, height=1,
                  text ="Monte Carlo Value:", font="verdana 10")
label_ins7_12d = Label(frame_menu_inserir12d, width=10, height=1,
                  text ="50", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins02d1 = Label(frame_menu_inserir12d, width=33, height=1,
                  text ="Input Shape - width:", font="verdana 10")
label_ins7_12d1 = Label(frame_menu_inserir12d, width=10, height=1,
                  text ="28", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins02d2 = Label(frame_menu_inserir12d, width=33, height=1,
                  text ="Input Shape - height:", font="verdana 10")
label_ins7_12d2 = Label(frame_menu_inserir12d, width=10, height=1,
                  text ="28", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins02d3 = Label(frame_menu_inserir12d, width=33, height=1,
                  text ="Input Shape - channels:", font="verdana 10")
label_ins7_12d3 = Label(frame_menu_inserir12d, width=10, height=1,
                  text ="1", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins02d31 = Label(frame_menu_inserir12d, width=33, height=1,
                  text ="Number of classes:", font="verdana 10")
label_ins7_12d31 = Label(frame_menu_inserir12d, width=10, height=1,
                  text ="10", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins112d = Label(frame_menu_inserir12d, width=10, height=1,
                  text ="                ", font="verdana 10")

texto_ins22d = Entry(frame_menu_inserir12d, width=8, font="verdana 9", justify=CENTER)
texto_ins22d.insert(END, '0.001')
texto_ins42d = Entry(frame_menu_inserir12d, width=8, font="verdana 9", justify=CENTER)
texto_ins42d.insert(END, '16')
texto_ins52d = Entry(frame_menu_inserir12d, width=8, font="verdana 9", justify=CENTER)
texto_ins52d.insert(END, '300')
texto_ins62d = Entry(frame_menu_inserir12d, width=8, font="verdana 9", justify=CENTER)
texto_ins62d.insert(END, '10')
texto_ins72d = Entry(frame_menu_inserir12d, width=8, font="verdana 9", justify=CENTER)
texto_ins72d.insert(END, '50')
texto_ins82d = Entry(frame_menu_inserir12d, width=8, font="verdana 9", justify=CENTER)
texto_ins82d.insert(END, '28')
texto_ins92d = Entry(frame_menu_inserir12d, width=8, font="verdana 9", justify=CENTER)
texto_ins92d.insert(END, '28')
texto_ins02d = Entry(frame_menu_inserir12d, width=8, font="verdana 9", justify=CENTER)
texto_ins02d.insert(END, '1')
texto_ins02d1 = Entry(frame_menu_inserir12d, width=8, font="verdana 9", justify=CENTER)
texto_ins02d1.insert(END, '10')

label_ins2d.grid(row=0, column=0)
label_ins02d.grid(row=0, column=1)
label_ins012d.grid(row=0, column=2)
label_ins22d.grid(row=2, column=0)
label_ins2_12d.grid(row=2, column=2)
label_ins42d.grid(row=4, column=0)
label_ins4_12d.grid(row=4, column=2)
label_ins52d.grid(row=5, column=0)
label_ins5_12d.grid(row=5, column=2)
label_ins62d.grid(row=6, column=0)
label_ins6_12d.grid(row=6, column=2)
label_ins72d.grid(row=7, column=0)
label_ins7_12d.grid(row=7, column=2)
label_ins02d1.grid(row=8, column=0)
label_ins02d2.grid(row=9, column=0)
label_ins02d3.grid(row=10, column=0)
label_ins02d31.grid(row=11, column=0)
label_ins7_12d1.grid(row=8, column=2)
label_ins7_12d2.grid(row=9, column=2)
label_ins7_12d3.grid(row=10, column=2)
label_ins7_12d31.grid(row=11, column=2)

texto_ins22d.grid(row=2,column=1)
texto_ins42d.grid(row=4,column=1)
texto_ins52d.grid(row=5,column=1)
texto_ins62d.grid(row=6,column=1)
texto_ins72d.grid(row=7,column=1)
texto_ins82d.grid(row=8,column=1)
texto_ins92d.grid(row=9,column=1)
texto_ins02d.grid(row=10,column=1)
texto_ins02d1.grid(row=11,column=1)
label_ins112d.grid(row=12,column=1)

cmd_verificavalores2d = Button(frame_menu_inserir12d, text="Insert/Save",
                             font=('verdana', 9, 'bold'),command=Insere2d)
cmd_verificavalores2d.grid(row=13,column=1)



#----------------------------------- Widgets Insert parameters 2 dimensons Regression------------------------------------

label_ins2dR = Label(frame_menu_inserir12dR, width=33, height=2,
                  text ="Insert Parameters for Regression", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins02dR = Label(frame_menu_inserir12dR, width=8, height=2,
                  text ="Values:", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins012dR = Label(frame_menu_inserir12dR, width=15, height=1,
                  text ="Recomended:", font=('verdana', 8, 'bold'), justify=CENTER)
label_ins22dR = Label(frame_menu_inserir12dR, width=33, height=1,
                  text ="Learning rate:", font="verdana 10")
label_ins2_12dR = Label(frame_menu_inserir12dR, width=10, height=1,
                  text ="0.001", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins42dR = Label(frame_menu_inserir12dR, width=33, height=1,
                  text ="Batch Size:", font="verdana 10")
label_ins4_12dR = Label(frame_menu_inserir12dR, width=10, height=1,
                  text ="16...128", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins52dR = Label(frame_menu_inserir12dR, width=33, height=1,
                  text ="Maximun number of epoch:", font="verdana 10")
label_ins5_12dR = Label(frame_menu_inserir12dR, width=10, height=1,
                  text ="300", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins62dR = Label(frame_menu_inserir12dR, width=33, height=1,
                  text ="Patience value:", font="verdana 10")
label_ins6_12dR = Label(frame_menu_inserir12dR, width=10, height=1,
                  text ="10", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins72dR = Label(frame_menu_inserir12dR, width=33, height=1,
                  text ="Monte Carlo Value:", font="verdana 10")
label_ins7_12dR = Label(frame_menu_inserir12dR, width=10, height=1,
                  text ="50", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins02d2Ri = Label(frame_menu_inserir12dR, width=33, height=1,
                  text ="Input Shape:", font="verdana 10")
label_ins7_12d31R = Label(frame_menu_inserir12dR, width=10, height=1,
                  text ="13", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins02d2Rir = Label(frame_menu_inserir12dR, width=33, height=1,
                  text ="Number of classes:", font="verdana 10")
label_ins7_12d31Rr = Label(frame_menu_inserir12dR, width=10, height=1,
                  text ="1", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins112dR = Label(frame_menu_inserir12dR, width=10, height=1,
                  text ="                ", font="verdana 10")

texto_ins22dR = Entry(frame_menu_inserir12dR, width=8, font="verdana 9", justify=CENTER)
texto_ins22dR.insert(END, '0.001')
texto_ins42dR = Entry(frame_menu_inserir12dR, width=8, font="verdana 9", justify=CENTER)
texto_ins42dR.insert(END, '16')
texto_ins52dR = Entry(frame_menu_inserir12dR, width=8, font="verdana 9", justify=CENTER)
texto_ins52dR.insert(END, '300')
texto_ins62dR = Entry(frame_menu_inserir12dR, width=8, font="verdana 9", justify=CENTER)
texto_ins62dR.insert(END, '10')
texto_ins72dR = Entry(frame_menu_inserir12dR, width=8, font="verdana 9", justify=CENTER)
texto_ins72dR.insert(END, '50')
texto_ins02d1R = Entry(frame_menu_inserir12dR, width=8, font="verdana 9", justify=CENTER)
texto_ins02d1R.insert(END, '13')
texto_ins02d1R1 = Entry(frame_menu_inserir12dR, width=8, font="verdana 9", justify=CENTER)
texto_ins02d1R1.insert(END, '1')

label_ins2dR.grid(row=0, column=0)
label_ins02dR.grid(row=0, column=1)
label_ins012dR.grid(row=0, column=2)
label_ins22dR.grid(row=2, column=0)
label_ins2_12dR.grid(row=2, column=2)
label_ins42dR.grid(row=4, column=0)
label_ins4_12dR.grid(row=4, column=2)
label_ins52dR.grid(row=5, column=0)
label_ins5_12dR.grid(row=5, column=2)
label_ins62dR.grid(row=6, column=0)
label_ins6_12dR.grid(row=6, column=2)
label_ins72dR.grid(row=7, column=0)
label_ins7_12dR.grid(row=7, column=2)
label_ins02d2Ri.grid(row=11, column=0)
label_ins02d2Rir.grid(row=12, column=0)
label_ins7_12d31R.grid(row=11, column=2)
label_ins7_12d31Rr.grid(row=12, column=2)


texto_ins22dR.grid(row=2,column=1)
texto_ins42dR.grid(row=4,column=1)
texto_ins52dR.grid(row=5,column=1)
texto_ins62dR.grid(row=6,column=1)
texto_ins72dR.grid(row=7,column=1)
texto_ins02d1R.grid(row=11,column=1)
texto_ins02d1R1.grid(row=12,column=1)
label_ins112dR.grid(row=13,column=1)

cmd_verificavalores2dR = Button(frame_menu_inserir12dR, text="Insert/Save",
                             font=('verdana', 9, 'bold'),command=Insere2dR)
cmd_verificavalores2dR.grid(row=13,column=1)

#------------------------------------------------------------ 
#Cancela o redimensionamento
menu_inicial.resizable(False, False)
menu_inicial.mainloop()
