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
tfd = tfp.distributions
tfpl = tfp.layers
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
from RunModel2d import *
from RunModel1d import *
from RunEditModel2d import *

#----------------------    Funções --------------------- 

#---- Ver informações sobre -----------
def semComando():
    hide_all_frames()
    frame_menuSobre.grid()

#---- Ver videos FAQ -----------
 
def verversao():
    hide_all_frames()
    frame_verversao.grid()
    
    
    
def runmodel_lenet_2d_call():
    runmodel_lenet_2d(learning_rate = float(texto_ins22d.get()),
                       batch_size = int(texto_ins42d.get()),
                       num_epochs = int(texto_ins52d.get()),
                       PATIENCE = int(texto_ins62d.get()),
                       num_monte_carlo = int(texto_ins72d.get())
                       )  
    return


def runmodel_lenet_1d_call():
    runmodel_lenet_1d(SAMPLING_FREQ = int(texto_ins1.get()),
                       EXAMINE_AVERAGE= int(texto_ins2.get()),
                       THRESHOLD_EARLY_STOPING = float(texto_ins4.get()),
                       BATCH_SIZE = int(texto_ins5.get()),
                       NUMBER_EPOCHS = int(texto_ins6.get()),
                       PATIENCE_VALUE = int(texto_ins7.get())
                       )
    return

#------ Function to runmodels Alexnet e Lenet for 2 dimensions ---------------   
 
def runmodel_alexlenet_2d_call():
    runmodel_alexnet_2d(learning_rate = float(texto_ins22d.get()),
                       batch_size = int(texto_ins42d.get()),
                       num_epochs = int(texto_ins52d.get()),
                       PATIENCE = int(texto_ins62d.get()),
                       num_monte_carlo = int(texto_ins72d.get())
                       )
    return    
    
def runmodel_lenet_2d_call():
    runmodel_lenet_2d(learning_rate = float(texto_ins22d.get()),
                       batch_size = int(texto_ins42d.get()),
                       num_epochs = int(texto_ins52d.get()),
                       PATIENCE = int(texto_ins62d.get()),
                       num_monte_carlo = int(texto_ins72d.get())
                       )
    return


#-------- imprime menu informacao ------------------
def vermenuinformacao():
    hide_all_frames()
    frame_menu_informacao.grid()
    
def vermenuinformacao1():
    hide_all_frames()
    frame_menu_informacao1.grid()

#------------ imprime menu inserir ------------
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
    
def menuinserir2():
    hide_all_frames()
    frame_menu_inserir2.grid()
    
def menuinserir2_2d():
    hide_all_frames()
    frame_menu_inserir2_2d.grid()

def menuinserir22d():
    hide_all_frames()
    frame_menu_inserir22d.grid()
    
def menuinserir3():
    hide_all_frames()
    frame_menu_inserir3.grid()
    
def menuinserir32d():
    hide_all_frames()
    frame_menu_inserir3.grid()

#--------------- imprime valores a correr ------------------------
    
def escolhemodelo():
    hide_all_frames()
    frame_menu_escolhemodelo.grid()
    escolhemodeloaux()


# --------------------   Funções para ver resultados -------------------------


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
    

    
# --------------------   Funções para editar camadas -------------------------
def aux1_insere_convolucao():
    hide_all_frames()
    frame_menu_convolucao.grid()
        
def aux1_insere_pooling():
    hide_all_frames()
    frame_menu_pooling.grid()

def aux1_insere_dense():
    hide_all_frames()
    frame_menu_dense.grid()


#-----------------  Limpa interface ------------------------------------------

def hide_all_frames():
    frame_menu_informacao.grid_forget()
    frame_menu_informacao1.grid_forget()
    frame_menu_inserir.grid_forget()
    frame_menu_inserir2d.grid_forget()
    frame_menu_inserir1.grid_forget()
    frame_menu_inserir12d.grid_forget()
    frame_menu_inserir2.grid_forget()
    frame_menu_inserir2_2d.grid_forget()
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
    
    
#--------------------- Insere valores ----------------------------------------
def Insere():
    SAMPLING_FREQ = int(texto_ins1.get())
    EXAMINE_AVERAGE= int(texto_ins2.get())
    THRESHOLD_EARLY_STOPING = float(texto_ins4.get())
    BATCH_SIZE = int(texto_ins5.get())
    NUMBER_EPOCHS = int(texto_ins6.get())
    PATIENCE_VALUE = int(texto_ins7.get())
    print("Parameters entered successfully")
    print("\nThe parameters that you insert are:")
    print("\nSignals sampling frequency:", SAMPLING_FREQ) 
    print("Times shoul the algoritm run:", EXAMINE_AVERAGE) 
    print("Epochs update relevant:", THRESHOLD_EARLY_STOPING) 
    print("Batch size:", BATCH_SIZE)  
    print("Maximum number of epoch:", NUMBER_EPOCHS) 
    print("Value for early stopping:", PATIENCE_VALUE) 
    menuinserir2()

def Insere2d():
    learning_rate = float(texto_ins22d.get())
    batch_size = int(texto_ins42d.get())
    num_epochs = int(texto_ins52d.get())
    PATIENCE = int(texto_ins62d.get())
    num_monte_carlo = int(texto_ins72d.get())
    print("Parameters entered successfully")
    print("Epochs update relevant:", learning_rate) 
    print("Batch size:", batch_size) 
    print("Maximum number of epoch:", num_epochs)  
    print("Value for early stopping:", PATIENCE) 
    print("Monte Carlo value:", num_monte_carlo) 
    menuinserir2_2d()
#-------------------   Combobox - Escolha de resultados do modelo AlexNet -------------------------------------------------------------

def genericresult2_aux():
    box_value3=StringVar()
    global escolhegeneric
    escolhegeneric = ttk.Combobox(frame_menu_vergeneric2, textvariable=box_value3, state='readonly')
    escolhegeneric["values"] = ["Acc-AlexNet MNIST","Auc-AlexNet MNIST","Npv-AlexNet MNIST","Ppv-AlexNet MNIST","Sen-AlexNet MNIST","Spe-AlexNet MNIST"]
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
    Button(frame_menu_vergeneric2,text="\nSee all the results in a plot", font=('verdana', 9, 'bold'), command=escolhe_resultado2_generic1).grid(row=1, column=1)
    Button(frame_menu_vergeneric2,text="\nSee each result in a plot", font=('verdana', 9, 'bold'), command=escolhe_resultado2_generic2).grid(row=3, column=1)
    Button(frame_menu_vergeneric2,text="\nPrint each result in Iptyton", font=('verdana', 9, 'bold'), command=escolhe_resultado2_generic).grid(row=4, column=1)

# ----------------------- Imprime os resultados da arquitetura Lenet ---------------

def escolhe_resultado2_generic1():
        with open('AlexNet2d\AccAlexNet2.txt', 'rb') as handle:
            a = pickle.load(handle)
        x = np.array(range(0, 10))
        y = np.array([a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]])
        with open('AlexNet2d\AucAlexNet2.txt', 'rb') as handle1:
            b = pickle.load(handle1)
        x = np.array(range(0, 10))
        y1 = np.array([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9]])       
        with open(rb'AlexNet2d\NpvAlexNet2.txt', 'rb') as handle2:
            c = pickle.load(handle2)
        x = np.array(range(0, 10))
        y2 = np.array([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]])       
        with open('AlexNet2d\PpvAlexNet2.txt', 'rb') as handle3:
            d = pickle.load(handle3)
        x = np.array(range(0, 10))
        y3 = np.array([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9]])     
        with open('AlexNet2d\SenAlexNet2.txt', 'rb') as handle4:
            e = pickle.load(handle4)
        x = np.array(range(0, 10))
        y4 = np.array([e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9]])
        with open('AlexNet2d\SpeAlexNet2.txt', 'rb') as handle5:
            f = pickle.load(handle5)
        x = np.array(range(0, 10))
        y5 = np.array([f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9]])
        plt.title("AlexNet ACC | ACU | NPV | PPV | SPE | SEN")
        plt.xlabel("Classes")
        plt.ylabel("ACC | ACU | NPV | PPV | SPE | SEN")
        plt.plot(x, y, y1)
        plt.plot(x, y2, y3)
        plt.plot(x, y4, y5)
        plt.legend()
        plt.show()


#---------------------------------------------------------------------


def escolhe_resultado2_generic2():
    if escolhegeneric.get() == "Acc-AlexNet MNIST":
        with open('AlexNet2d\AccAlexNet2.txt', 'rb') as handle:
            a = pickle.load(handle)
        x = np.array(range(0, 10))
        y = np.array([a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]])
        plt.title("Acc values")
        plt.xlabel("Classes")
        plt.ylabel("Acc values")
        plt.plot(x, y, color = "Orange", marker = "o", label = "Acc values")
        plt.legend()
        plt.show()
    
    elif escolhegeneric.get() == "Auc-AlexNet MNIST":
        with open('AlexNet2d\AucAlexNet2.txt', 'rb') as handle1:
            b = pickle.load(handle1)
        x = np.array(range(0, 10))
        y1 = np.array([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9]])
        plt.title("AlexNet - Auc values")
        plt.xlabel("Classes")
        plt.ylabel("Auc values")
        plt.plot(x, y1, color = "Green", marker = "o", label = "Auc values")
        plt.legend()
        plt.show()
        
        
    elif escolhegeneric.get() == "Npv-AlexNet MNIST":
        with open(rb'AlexNet2d\NpvAlexNet2.txt', 'rb') as handle2:
            c = pickle.load(handle2)
        x = np.array(range(0, 10))
        y2 = np.array([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]])
        plt.title("NPV values")
        plt.xlabel("Classes")
        plt.ylabel("NPV values")
        plt.plot(x, y2, color = "blue", marker = "o", label = "NPV values")
        plt.legend()
        plt.show()
        
    elif escolhegeneric.get() == "Ppv-AlexNet MNIST":
        with open('AlexNet2d\PpvAlexNet2.txt', 'rb') as handle3:
            d = pickle.load(handle3)
        x = np.array(range(0, 10))
        y3 = np.array([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9]])
        plt.title("AlexNet - PPV valuess")
        plt.xlabel("Classes")
        plt.ylabel("PPV values")
        plt.plot(x, y3, color = "red", marker = "o", label = "PPV values")
        plt.legend()
        plt.show()
      
    elif escolhegeneric.get() == "Sen-AlexNet MNIST":
        with open('AlexNet2d\SenAlexNet2.txt', 'rb') as handle4:
            e = pickle.load(handle4)
        x = np.array(range(0, 10))
        y4 = np.array([e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9]])
        plt.title("AlexNet - Sen")
        plt.xlabel("Classes")
        plt.ylabel("Sen values")
        plt.plot(x, y4, color = "black", marker = "o", label = "Sen values")
        plt.legend()
        plt.show()

    elif escolhegeneric.get() == "Spe-AlexNet MNIST":
        with open('AlexNet2d\SpeAlexNet2.txt', 'rb') as handle5:
            f = pickle.load(handle5)
        x = np.array(range(0, 10))
        y5 = np.array([f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9]])
        plt.title("AlexNet - Spe")
        plt.xlabel("Classes")
        plt.ylabel("Spe values")
        plt.plot(x, y5, color = "Purple", marker = "o", label = "Spe values")
        plt.legend()
        plt.show()



def escolhe_resultado2_generic():
    if escolhegeneric.get() == "Acc-AlexNet MNIST":
        object_Acc_Generic = []
        with (open("AlexNet2d\AccAlexNet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_Acc_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_Acc_Generic)   
                 
    elif escolhegeneric.get() == "Auc-AlexNet MNIST":
        object_Auc_Generic = []
        with (open("AlexNet2d\AucAlexNet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_Auc_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_Auc_Generic)
                 
    elif escolhegeneric.get() == "Npv-AlexNet MNIST":
        object_NPV_Generic = []
        with (open(rb"AlexNet2d\NpvAlexNet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_NPV_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_NPV_Generic)
                 
    elif escolhegeneric.get() == "Ppv-AlexNet MNIST":
        object_PPV_Generic = []
        with (open("AlexNet2d\PpvAlexNet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_PPV_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_PPV_Generic)
                 
    elif escolhegeneric.get() == "Sen-AlexNet MNIST":
        object_SEN_Generic = []
        with (open("AlexNet2d\SenAlexNet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_SEN_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_SEN_Generic)

    elif escolhegeneric.get() == "Spe-AlexNet MNIST":
        object_SPE_Generic = []
        with (open("AlexNet2d\SpeAlexNet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_SPE_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_SPE_Generic)

    else:    
            print("Select the result in the select box") 
    
#-------------------   Combobox - Escolha de resultados do modelo leNet -------------------------------------------------------------

def genericresult1_aux():
    box_value3=StringVar()
    global escolhegeneric
    escolhegeneric = ttk.Combobox(frame_menu_vergeneric1, textvariable=box_value3, state='readonly')
    escolhegeneric["values"] = ["Acc-LeNet MNIST","Auc-LeNet MNIST","Npv-LeNet MNIST","Ppv-LeNet MNIST","Sen-LeNet MNIST","Spe-LeNet MNIST"]
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
    Button(frame_menu_vergeneric1,text="\nSee all the results in a plot", font=('verdana', 9, 'bold'), command=escolhe_resultado1_generic1).grid(row=1, column=1)
    Button(frame_menu_vergeneric1,text="\nSee each result in a plot", font=('verdana', 9, 'bold'), command=escolhe_resultado1_generic2).grid(row=3, column=1)
    Button(frame_menu_vergeneric1,text="\nPrint each result in Iptyton", font=('verdana', 9, 'bold'), command=escolhe_resultado1_generic).grid(row=4, column=1)

# ----------------------- Imprime os resultados da arquitetura Lenet ---------------

def escolhe_resultado1_generic1():
        with open('LeNet2d\AccLeNet2.txt', 'rb') as handle:
            a = pickle.load(handle)
        x = np.array(range(0, 10))
        y = np.array([a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]])
        with open('LeNet2d\AucLeNet2.txt', 'rb') as handle1:
            b = pickle.load(handle1)
        x = np.array(range(0, 10))
        y1 = np.array([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9]])       
        with open(rb'LeNet2d\NpvLeNet2.txt', 'rb') as handle2:
            c = pickle.load(handle2)
        x = np.array(range(0, 10))
        y2 = np.array([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]])       
        with open('LeNet2d\PpvLeNet2.txt', 'rb') as handle3:
            d = pickle.load(handle3)
        x = np.array(range(0, 10))
        y3 = np.array([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9]])     
        with open('LeNet2d\SenLeNet2.txt', 'rb') as handle4:
            e = pickle.load(handle4)
        x = np.array(range(0, 10))
        y4 = np.array([e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9]])
        with open('LeNet2d\SpeLeNet2.txt', 'rb') as handle5:
            f = pickle.load(handle5)
        x = np.array(range(0, 10))
        y5 = np.array([f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9]])
        plt.title("LeNet - ACC | ACU | NPV | PPV | SPE | SEN")
        plt.xlabel("Classes")
        plt.ylabel("ACC | ACU | NPV | PPV | SPE | SEN")
        plt.plot(x, y, y1)
        plt.plot(x, y2, y3)
        plt.plot(x, y4, y5)
        plt.legend()
        plt.show()


#---------------------------------------------------------------------


def escolhe_resultado1_generic2():
    if escolhegeneric.get() == "Acc-LeNet MNIST":
        with open('LeNet2d\AccLeNet2.txt', 'rb') as handle:
            a = pickle.load(handle)
        x = np.array(range(0, 10))
        y = np.array([a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]])
        plt.title("LeNet - Acc values")
        plt.xlabel("Classes")
        plt.ylabel("Acc values")
        plt.plot(x, y, color = "Orange", marker = "o", label = "Acc values")
        plt.legend()
        plt.show()
    
    elif escolhegeneric.get() == "Auc-LeNet MNIST":
        with open('LeNet2d\AucLeNet2.txt', 'rb') as handle1:
            b = pickle.load(handle1)
        x = np.array(range(0, 10))
        y1 = np.array([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9]])
        plt.title("LeNet - Auc values")
        plt.xlabel("Classes")
        plt.ylabel("Auc values")
        plt.plot(x, y1, color = "Green", marker = "o", label = "Auc values")
        plt.legend()
        plt.show()
        
        
    elif escolhegeneric.get() == "Npv-LeNet MNIST":
        with open(rb'LeNet2d\NpvLeNet2.txt', 'rb') as handle2:
            c = pickle.load(handle2)
        x = np.array(range(0, 10))
        y2 = np.array([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]])
        plt.title("LeNet - NPV values")
        plt.xlabel("Classes")
        plt.ylabel("NPV values")
        plt.plot(x, y2, color = "blue", marker = "o", label = "NPV values")
        plt.legend()
        plt.show()
        
    elif escolhegeneric.get() == "Ppv-LeNet MNIST":
        with open('LeNet2d\PpvLeNet2.txt', 'rb') as handle3:
            d = pickle.load(handle3)
        x = np.array(range(0, 10))
        y3 = np.array([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9]])
        plt.title("LeNet - PPV values")
        plt.xlabel("Classes")
        plt.ylabel("PPV values")
        plt.plot(x, y3, color = "red", marker = "o", label = "PPV values")
        plt.legend()
        plt.show()
      
    elif escolhegeneric.get() == "Sen-LeNet MNIST":
        with open('LeNet2d\SenLeNet2.txt', 'rb') as handle4:
            e = pickle.load(handle4)
        x = np.array(range(0, 10))
        y4 = np.array([e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9]])
        plt.title("LeNet - Sen values")
        plt.xlabel("Classes")
        plt.ylabel("Sen values")
        plt.plot(x, y4, color = "black", marker = "o", label = "Sen values")
        plt.legend()
        plt.show()

    elif escolhegeneric.get() == "Spe-LeNet MNIST":
        with open('LeNet2d\SpeLeNet2.txt', 'rb') as handle5:
            f = pickle.load(handle5)
        x = np.array(range(0, 10))
        y5 = np.array([f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9]])
        plt.title("LeNet - Spe values")
        plt.xlabel("Classes")
        plt.ylabel("Spe values")
        plt.plot(x, y5, color = "Purple", marker = "o", label = "Spe values")
        plt.legend()
        plt.show()



def escolhe_resultado1_generic():
    if escolhegeneric.get() == "Acc-LeNet MNIST":
        object_Acc_Generic = []
        with (open("LeNet2d\AccLeNet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_Acc_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_Acc_Generic)   
                 
    elif escolhegeneric.get() == "Auc-LeNet MNIST":
        object_Auc_Generic = []
        with (open("LeNet2d\AucLeNet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_Auc_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_Auc_Generic)
                 
    elif escolhegeneric.get() == "Npv-LeNet MNIST":
        object_NPV_Generic = []
        with (open(rb"LeNet2d\NpvLeNet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_NPV_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_NPV_Generic)
                 
    elif escolhegeneric.get() == "Ppv-LeNet MNIST":
        object_PPV_Generic = []
        with (open("LeNet2d\PpvLeNet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_PPV_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_PPV_Generic)
                 
    elif escolhegeneric.get() == "Sen-LeNet MNIST":
        object_SEN_Generic = []
        with (open("LeNet2d\SenLeNet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_SEN_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_SEN_Generic)

    elif escolhegeneric.get() == "Spe-LeNet MNIST":
        object_SPE_Generic = []
        with (open("LeNet2d\SpeLeNet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_SPE_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_SPE_Generic)

    else:    
            print("Select the result in the select box") 
            

#-------------------   Combobox - Escolha de resultados do modelo leNet -------------------------------------------------------------

def genericresult_aux():
    box_value3=StringVar()
    global escolhegeneric
    escolhegeneric = ttk.Combobox(frame_menu_vergeneric, textvariable=box_value3, state='readonly')
    escolhegeneric["values"] = ["Acc-LeNet ISRUC-SLEEP","Auc-LeNet ISRUC-SLEEP","Npv-LeNet ISRUC-SLEEP","Ppv-LeNet ISRUC-SLEEP","Sen-LeNet ISRUC-SLEEP","Spe-LeNet ISRUC-SLEEP"]
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
    Button(frame_menu_vergeneric,text="\nSee all the results in a plot", font=('verdana', 9, 'bold'), command=escolhe_resultado_generic1).grid(row=1, column=1)
    Button(frame_menu_vergeneric,text="\nSee each result in a plot", font=('verdana', 9, 'bold'), command=escolhe_resultado_generic2).grid(row=3, column=1)
    Button(frame_menu_vergeneric,text="\nPrint each result in Iptyton", font=('verdana', 9, 'bold'), command=escolhe_resultado_generic).grid(row=4, column=1)


# ----------------------- Imprime os resultados da arquitetura Lenet ---------------

def escolhe_resultado_generic1():
    object_NPV_Generic = []
    with (open(rb"LeNet1d\NpvLeNet1.txt","rb")) as openfile:
          while True:
             try:
                object_NPV_Generic.append(pickle.load(openfile))
             except EOFError:
                 break
    list_NPV_all_epochs = []
    for epoch in object_NPV_Generic:
        media = 0
        for i in epoch:
            media += epoch[i]
        media /= 5
        list_NPV_all_epochs.append(media)
 #---------------------------------------       
    object_PPV_Generic = []
    with (open("LeNet1d\PpvLeNet1.txt","rb")) as openfile:
          while True:
             try:
                object_PPV_Generic.append(pickle.load(openfile))
             except EOFError:
                 break
    list_PPV_all_epochs = []
    for epoch in object_PPV_Generic:
        media = 0
        for i in epoch:
            media += epoch[i]
        media /= 5
        list_PPV_all_epochs.append(media) 
 #---------------------------------------------    
    object_ACC_Generic = []
    with (open("LeNet1d\AccLeNet1.txt","rb")) as openfile:
          while True:
             try:
                object_ACC_Generic.append(pickle.load(openfile))
             except EOFError:
                 break
    list_ACC_all_epochs = []
    for epoch in object_ACC_Generic:
        media = 0
        for i in epoch:
            media += epoch[i]
        media /= 5
        list_ACC_all_epochs.append(media)    
#-----------------------------------------------------        
    object_AUC_Generic = []
    with (open("LeNet1d\AucLeNet1.txt","rb")) as openfile:
          while True:
             try:
                object_AUC_Generic.append(pickle.load(openfile))
             except EOFError:
                 break
    list_AUC_all_epochs = []
    for epoch in object_AUC_Generic:
        media = 0
        for i in epoch:
            media += epoch[i]
        media /= 5
        list_AUC_all_epochs.append(media)    
 #---------------------------------------------------------------       
    object_SEN_Generic = []
    with (open("LeNet1d\SenLeNet1.txt","rb")) as openfile:
          while True:
             try:
                object_SEN_Generic.append(pickle.load(openfile))
             except EOFError:
                 break
    list_SEN_all_epochs = []
    for epoch in object_SEN_Generic:
        media = 0
        for i in epoch:
            media += epoch[i]
        media /= 5
        list_SEN_all_epochs.append(media)  
#------------------------------------------------------------------         
    object_SPE_Generic = []
    with (open("LeNet1d\SpeLeNet1.txt","rb")) as openfile:
          while True:
             try:
                object_SPE_Generic.append(pickle.load(openfile))
             except EOFError:
                 break
    list_SPE_all_epochs = []
    for epoch in object_SPE_Generic:
        media = 0
        for i in epoch:
            media += epoch[i]
        media /= 5
        list_SPE_all_epochs.append(media)      
#---------------------------------------------------------------------
    xpoints = range(len(object_NPV_Generic))  
    plt.figure()
    plt.plot(xpoints, list_NPV_all_epochs, list_PPV_all_epochs)
    plt.plot(xpoints, list_ACC_all_epochs, list_AUC_all_epochs)
    plt.plot(xpoints, list_SEN_all_epochs, list_SPE_all_epochs)
    plt.title("LeNet - ACC | AUC | NPV | PPV | SPE | SEN")
    plt.ylabel('ACC | AUC | NPV | PPV | SPE | SEN')
    plt.show()


def escolhe_resultado_generic2_cm():
        object_m_Generic = []
        with (open("LeNet1d\all_metrics.txt","rb")) as openfile:
              while True:
                 try:
                    object_m_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
        list_ACC_all_epochs = []
        for epoch in object_m_Generic:
            media = 0
            for i in epoch:
                media += epoch[i]
            media /= 5
            list_ACC_all_epochs.append(media)  
        xpoints = range(len(object_m_Generic))  
        plt.figure()
        plt.plot(xpoints, list_ACC_all_epochs)
        plt.xlabel('Number of epochs')
        plt.ylabel('Confusion Matrix')
        plt.show()
       

def escolhe_resultado_generic2():
    if escolhegeneric.get() == "Acc-LeNet ISRUC-SLEEP":
        object_ACC_Generic = []
        with (open("LeNet1d\AccLeNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_ACC_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
        list_ACC_all_epochs = []
        for epoch in object_ACC_Generic:
            media = 0
            for i in epoch:
                media += epoch[i]
            media /= 5
            list_ACC_all_epochs.append(media)  
        xpoints = range(len(object_ACC_Generic))  
        plt.figure()
        plt.plot(xpoints, list_ACC_all_epochs)
        plt.title("LeNet - ACC")
        plt.ylabel('Value of Acc')
        plt.show()
       
        
    elif escolhegeneric.get() == "Auc-LeNet ISRUC-SLEEP":
        object_AUC_Generic = []
        with (open("LeNet1d\AucLeNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_AUC_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
        list_ACC_all_epochs = []
        for epoch in object_AUC_Generic:
            media = 0
            for i in epoch:
                media += epoch[i]
            media /= 5
            list_ACC_all_epochs.append(media)  
        xpoints = range(len(object_AUC_Generic))  
        plt.figure()
        plt.title("LeNet - AUC")
        plt.plot(xpoints, list_ACC_all_epochs)
        plt.ylabel('Value of AUC')
        plt.show()
        
        
    elif escolhegeneric.get() == "Npv-LeNet ISRUC-SLEEP":
        object_NPV_Generic = []
        with (open(rb"LeNet1d\NpvLeNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_NPV_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
        list_ACC_all_epochs = []
        for epoch in object_NPV_Generic:
            media = 0
            for i in epoch:
                media += epoch[i]
            media /= 5
            list_ACC_all_epochs.append(media)  
        xpoints = range(len(object_NPV_Generic))  
        plt.figure()
        plt.plot(xpoints, list_ACC_all_epochs)
        plt.title("LeNet - NPV")
        plt.ylabel('Value of NPV')
        plt.show()
        
        
    elif escolhegeneric.get() == "Ppv-LeNet ISRUC-SLEEP":
        object_PPV_Generic = []
        with (open("LeNet1d\PpvLeNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_PPV_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
        list_ACC_all_epochs = []
        for epoch in object_PPV_Generic:
            media = 0
            for i in epoch:
                media += epoch[i]
            media /= 5
            list_ACC_all_epochs.append(media)  
    #---------------------------------------------------------------------
        xpoints = range(len(object_PPV_Generic))  
        plt.figure()
        plt.plot(xpoints, list_ACC_all_epochs)
        plt.xlabel('Number of epochs')
        plt.title("LeNet - PPV")
        plt.ylabel('Value of PPV')
        plt.show()
        
    elif escolhegeneric.get() == "Sen-LeNet ISRUC-SLEEP":
        object_SEN_Generic = []
        with (open("LeNet1d\SenLeNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_SEN_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
        list_ACC_all_epochs = []
        for epoch in object_SEN_Generic:
            media = 0
            for i in epoch:
                media += epoch[i]
            media /= 5
            list_ACC_all_epochs.append(media)  
        xpoints = range(len(object_SEN_Generic))  
        plt.figure()
        plt.plot(xpoints, list_ACC_all_epochs)
        plt.title("LeNet - SEN")
        plt.ylabel('Value of Sen')
        plt.show()
        
        
    elif escolhegeneric.get() == "Spe-LeNet ISRUC-SLEEP":
        object_SPE_Generic = []
        with (open("LeNet1d\SpeLeNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_SPE_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
        list_ACC_all_epochs = []
        for epoch in object_SPE_Generic:
            media = 0
            for i in epoch:
                media += epoch[i]
            media /= 5
            list_ACC_all_epochs.append(media)  
        xpoints = range(len(object_SPE_Generic))  
        plt.figure()
        plt.plot(xpoints, list_ACC_all_epochs)
        plt.title("LeNet - SPE")
        plt.ylabel('Value of Spe')
        plt.show()


def escolhe_resultado_generic():
    
    if escolhegeneric.get() == "Acc-Alexnet ISRUC-SLEEP":
        object_Acc_Generic = []
        with (open("LeNet1d\AccLeNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_Acc_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_Acc_Generic)   
                 
    elif escolhegeneric.get() == "Auc-Alexnet ISRUC-SLEEP":
        object_Auc_Generic = []
        with (open("LeNet1d\AucLeNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_Auc_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_Auc_Generic)
                 
    elif escolhegeneric.get() == "Npv-Alexnet ISRUC-SLEEP":
        object_NPV_Generic = []
        with (open(r"LeNet1d\NpvLeNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_NPV_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_NPV_Generic)
                 
    elif escolhegeneric.get() == "Ppv-Alexnet ISRUC-SLEEP":
        object_PPV_Generic = []
        with (open("LeNet1d\PpvLeNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_PPV_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_PPV_Generic)
                 
    elif escolhegeneric.get() == "Sen-Alexnet ISRUC-SLEEP":
        object_SEN_Generic = []
        with (open("LeNet1d\SenLeNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_SEN_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_SEN_Generic)

    elif escolhegeneric.get() == "Spe-Alexnet ISRUC-SLEEP":
        object_SPE_Generic = []
        with (open("LeNet1d\SpeLeNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_SPE_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_SPE_Generic)

    else:    
            print("Select the result in the select box") 
            
#-------------------   Combobox - Escolha de resultados do modelo leNet -------------------------------------------------------------

def alexnetresult_aux():
    box_value3=StringVar()
    global escolhegeneric1
    escolhegeneric1 = ttk.Combobox(frame_menu_veralexnet, textvariable=box_value3, state='readonly')
    escolhegeneric1["values"] = ["Acc-AlexNet ISRUC-SLEEP","Auc-AlexNet ISRUC-SLEEP","Npv-AlexNet ISRUC-SLEEP","Ppv-AlexNet ISRUC-SLEEP","Sen-AlexNet ISRUC-SLEEP","Spe-AlexNet ISRUC-SLEEP"]
    escolhegeneric1.current(0)
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
    escolhegeneric1.grid(row=3, column=0)
    Button(frame_menu_veralexnet,text="\nSee all the results in a plot", font=('verdana', 9, 'bold'), command=alexnet_resultado_generic1).grid(row=1, column=1)
    Button(frame_menu_veralexnet,text="\nSee each result in a plot", font=('verdana', 9, 'bold'), command=alexnet_resultado_generic2).grid(row=3, column=1)
    Button(frame_menu_veralexnet,text="\nPrint each result in Iptyton", font=('verdana', 9, 'bold'), command=alexnet_resultado_generic).grid(row=4, column=1)

# ----------------------- Imprime os resultados da arquitetura Alexnet ---------------

def alexnet_resultado_generic1():
    object_NPV_alexnet = []
    with (open(rb"AlexNet1d\NpvAlexNet1.txt","rb")) as openfile:
          while True:
             try:
                object_NPV_alexnet.append(pickle.load(openfile))
             except EOFError:
                 break
    list_NPV_all_epochs = []
    for epoch in object_NPV_alexnet:
        media = 0
        for i in epoch:
            media += epoch[i]
        media /= 5
        list_NPV_all_epochs.append(media)
 #---------------------------------------       
    object_PPV_alexnet = []
    with (open("AlexNet1d\PpvAlexNet1.txt","rb")) as openfile:
          while True:
             try:
                object_PPV_alexnet.append(pickle.load(openfile))
             except EOFError:
                 break
    list_PPV_all_epochs = []
    for epoch in object_PPV_alexnet:
        media = 0
        for i in epoch:
            media += epoch[i]
        media /= 5
        list_PPV_all_epochs.append(media) 
 #---------------------------------------------    
    object_ACC_alexnet = []
    with (open("AlexNet1d\AccAlexNet1.txt","rb")) as openfile:
          while True:
             try:
                object_ACC_alexnet.append(pickle.load(openfile))
             except EOFError:
                 break
    list_ACC_all_epochs = []
    for epoch in object_ACC_alexnet:
        media = 0
        for i in epoch:
            media += epoch[i]
        media /= 5
        list_ACC_all_epochs.append(media)    
#-----------------------------------------------------        
    object_AUC_alexnet = []
    with (open("AlexNet1d\AucAlexNet1.txt","rb")) as openfile:
          while True:
             try:
                object_AUC_alexnet.append(pickle.load(openfile))
             except EOFError:
                 break
    list_AUC_all_epochs = []
    for epoch in object_AUC_alexnet:
        media = 0
        for i in epoch:
            media += epoch[i]
        media /= 5
        list_AUC_all_epochs.append(media)    
 #---------------------------------------------------------------       
    object_SEN_alexnet = []
    with (open("AlexNet1d\SenAlexNet1.txt","rb")) as openfile:
          while True:
             try:
                object_SEN_alexnet.append(pickle.load(openfile))
             except EOFError:
                 break
    list_SEN_all_epochs = []
    for epoch in object_SEN_alexnet:
        media = 0
        for i in epoch:
            media += epoch[i]
        media /= 5
        list_SEN_all_epochs.append(media)  
#------------------------------------------------------------------         
    object_SPE_alexnet = []
    with (open("AlexNet1d\SpeAlexNet1.txt","rb")) as openfile:
          while True:
             try:
                object_SPE_alexnet.append(pickle.load(openfile))
             except EOFError:
                 break
    list_SPE_all_epochs = []
    for epoch in object_SPE_alexnet:
        media = 0
        for i in epoch:
            media += epoch[i]
        media /= 5
        list_SPE_all_epochs.append(media)      
#---------------------------------------------------------------------
    xpoints = range(len(object_NPV_alexnet))  
    plt.figure()
    plt.plot(xpoints, list_NPV_all_epochs, list_PPV_all_epochs)
    plt.plot(xpoints, list_ACC_all_epochs, list_AUC_all_epochs)
    plt.plot(xpoints, list_SEN_all_epochs, list_SPE_all_epochs)
    plt.title("AlexNet - ACC | AUC | NPV | PPV | SPE | SEN")
    plt.ylabel('ACC | AUC | NPV | PPV | SPE | SEN')
    plt.show()

def alexnet_resultado_generic2():
    if escolhegeneric1.get() == "Acc-AlexNet ISRUC-SLEEP":
        object_ACC_alexnet = []
        with (open("AlexNet1d\AccAlexNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_ACC_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
        list_ACC_all_epochs = []
        for epoch in object_ACC_alexnet:
            media = 0
            for i in epoch:
                media += epoch[i]
            media /= 5
            list_ACC_all_epochs.append(media)  
        xpoints = range(len(object_ACC_alexnet))  
        plt.figure()
        plt.plot(xpoints, list_ACC_all_epochs)
        plt.title("AlexNet - ACC ")
        plt.xlabel('Number of epochs')
        plt.ylabel('Value of Acc')
        plt.show()
       
        
    elif escolhegeneric1.get() == "Auc-AlexNet ISRUC-SLEEP":
        object_AUC_alexnet = []
        with (open("AlexNet1d\AucAlexNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_AUC_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
        list_ACC_all_epochs = []
        for epoch in object_AUC_alexnet:
            media = 0
            for i in epoch:
                media += epoch[i]
            media /= 5
            list_ACC_all_epochs.append(media)  
        xpoints = range(len(object_AUC_alexnet))  
        plt.figure()
        plt.plot(xpoints, list_ACC_all_epochs)
        plt.title("AlexNet - AUC ")
        plt.ylabel('Value of AUC')
        plt.show()
        
        
    elif escolhegeneric1.get() == "Npv-AlexNet ISRUC-SLEEP":
        object_NPV_alexnet = []
        with (open(rb"AlexNet1d\NpvAlexNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_NPV_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
        list_ACC_all_epochs = []
        for epoch in object_NPV_alexnet:
            media = 0
            for i in epoch:
                media += epoch[i]
            media /= 5
            list_ACC_all_epochs.append(media)  
        xpoints = range(len(object_NPV_alexnet))  
        plt.figure()
        plt.plot(xpoints, list_ACC_all_epochs)
        plt.title("AlexNet - NPV ")
        plt.ylabel('Value of NPV')
        plt.show()
        
        
    elif escolhegeneric1.get() == "Ppv-AlexNet ISRUC-SLEEP":
        object_PPV_alexnet = []
        with (open("AlexNet1d\PpvAlexNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_PPV_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
        list_ACC_all_epochs = []
        for epoch in object_PPV_alexnet:
            media = 0
            for i in epoch:
                media += epoch[i]
            media /= 5
            list_ACC_all_epochs.append(media)  
    #---------------------------------------------------------------------
        xpoints = range(len(object_PPV_alexnet))  
        plt.figure()
        plt.plot(xpoints, list_ACC_all_epochs)
        plt.title("AlexNet - PPV ")
        plt.ylabel('Value of PPV')
        plt.show()
        
    elif escolhegeneric1.get() == "Sen-AlexNet ISRUC-SLEEP":
        object_SEN_alexnet = []
        with (open("AlexNet1d\SenAlexNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_SEN_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
        list_ACC_all_epochs = []
        for epoch in object_SEN_alexnet:
            media = 0
            for i in epoch:
                media += epoch[i]
            media /= 5
            list_ACC_all_epochs.append(media)  
        xpoints = range(len(object_SEN_alexnet))  
        plt.figure()
        plt.plot(xpoints, list_ACC_all_epochs)
        plt.title("AlexNet - SEN")
        plt.ylabel('Value of Sen')
        plt.show()
        
        
    elif escolhegeneric1.get() == "Spe-AlexNet ISRUC-SLEEP":
        object_SPE_alexnet = []
        with (open("AlexNet1d\SpeAlexNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_SPE_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
        list_ACC_all_epochs = []
        for epoch in object_SPE_alexnet:
            media = 0
            for i in epoch:
                media += epoch[i]
            media /= 5
            list_ACC_all_epochs.append(media)  
        xpoints = range(len(object_SPE_alexnet))  
        plt.figure()
        plt.plot(xpoints, list_ACC_all_epochs)
        plt.title("AlexNet - SPE")
        plt.ylabel('Value of Spe')
        plt.show()


def alexnet_resultado_generic():
    
    if escolhegeneric1.get() == "Acc-AlexNet ISRUC-SLEEP":
        object_Acc_alexnet = []
        with (open("AlexNet1d\AccAlexNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_Acc_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_Acc_alexnet)   
                 
    elif escolhegeneric1.get() == "Auc-AlexNet ISRUC-SLEEP":
        object_Auc_alexnet = []
        with (open("AlexNet1d\AucAlexNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_Auc_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_Auc_alexnet)
                 
    elif escolhegeneric1.get() == "Npv-AlexNet ISRUC-SLEEP":
        object_NPV_alexnet = []
        with (open(rb"AlexNet1d\NpvAlexNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_NPV_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_NPV_alexnet)
                 
    elif escolhegeneric1.get() == "Ppv-AlexNet ISRUC-SLEEP":
        object_PPV_alexnet = []
        with (open("AlexNet1d\PpvAlexNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_PPV_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_PPV_alexnet)
                 
    elif escolhegeneric1.get() == "Sen-AlexNet ISRUC-SLEEP":
        object_SEN_alexnet = []
        with (open("AlexNet1d\SenAlexNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_SEN_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_SEN_alexnet)

    elif escolhegeneric1.get() == "Spe-AlexNet ISRUC-SLEEP":
        object_SPE_alexnet = []
        with (open("AlexNet1d\SpeAlexNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_SPE_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_SPE_alexnet)

    else:    
            print("Select the result in the select box") 
 
 
    
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

#------------------- Cria Frames ---------------------------------------------

frame_menuSobre = Frame(menu_inicial)
frame_verversao = Frame(menu_inicial)
frame_menu_informacao = Frame(menu_inicial)
frame_menu_informacao1 = Frame(menu_inicial)
frame_menu_inserir =  Frame(menu_inicial)
frame_menu_inserir1 =  Frame(menu_inicial)
frame_menu_inserir2 =  Frame(menu_inicial)
frame_menu_inserir2_2d =  Frame(menu_inicial)
frame_menu_inserir3 =  Frame(menu_inicial)
frame_menu_inserir2d =  Frame(menu_inicial)
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

#---------------------------- Gui Barra de Menus -------------------------------

barraDeMenus = Menu(menu_inicial)
menuinformacao = Menu(barraDeMenus,tearoff=0)
menuinformacao.add_command(label="See how to Start 1 Dimensions Datasets",command=vermenuinformacao)
menuinformacao.add_command(label="See how to Start 2 Dimensions Datasets",command=vermenuinformacao1)
barraDeMenus.add_cascade(label="Start",menu=menuinformacao)

#------------------------       Separador Insere          -------------------
menuinsere = Menu(barraDeMenus,tearoff=0)
menuinsere.add_command(label="1 Dimension parameters",command=menuinserir1)
menuinsere.add_command(label="2 Dimensions parameters",command=menuinserir12d)
menuinsere.add_separator()
barraDeMenus.add_cascade(label="Insert | Run",menu=menuinsere)

#------------------------       Separador Resultados      -------------------
menuresultados = Menu(barraDeMenus,tearoff=0)
menuresultados.add_command(label="1 Dimnension - LeNet",command=genericresult)
menuresultados.add_command(label="1 Dimension - AlexNet",command=alexnetresult)
menuresultados.add_command(label="2 Dimensions - LeNet",command=genericresult1)
menuresultados.add_command(label="2 Dimension - AlexNet",command=genericresult2)
menuresultados.add_command(label="Edit 1 Dimension",command=aux1_insere_convolucao)
menuresultados.add_command(label="Edit 2 Dimensions",command=aux1_insere_convolucao)
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



#---------------------------- Menu inicial IRUC DATASET ------------- 
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

#sympy==1.8  keras==2.6.0 matplotlib = 3.3.4 Phyton = 3.8.8
label_mensagem_versao.grid(row=0, column=0)
label_mensagem1_versao.grid(row=1, column=0)
label_mensagem2_versao.grid(row=2, column=0)
label_mensagem3_versao.grid(row=3, column=0)
label_mensagem4_versao.grid(row=4, column=0)
label_mensagem5_versao.grid(row=5, column=0)


#---------------------------- Menu inicial IRUC DATASET ------------- 
label_mensagem_sobre = Label(frame_menuSobre,
                  text ="\n\n          Graphical Interface for development of CNN Probabilistics", font="verdana 10 bold")
label_mensagem1_sobre = Label(frame_menuSobre,
                  text ="\n Program developed by:", font="verdana 10")
label_mensagem2_sobre = Label(frame_menuSobre,
                  text ="\n     Aníbal João Lopes Chaves", font="verdana 10 bold")
label_mensagem21_sobre = Label(frame_menuSobre,
                  text ="\n June of 2022", font="verdana 10")


label_mensagem_sobre.grid(row=0, column=0)
label_mensagem1_sobre.grid(row=1, column=0)
label_mensagem2_sobre.grid(row=2, column=0)
label_mensagem21_sobre.grid(row=3, column=0)
#---------------------------- Menu inicial IRUC DATASET --------------
    

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

#---------------------------- Menu inicial IRUC MNIST --------------
    
label_mensagem3_MNIST = Label(frame_menu_informacao1,
                  text ="\n\n              1 - Select | Run Parameters and Arquitectures.", font="verdana 10 bold")
label_mensagem4_MNIST = Label(frame_menu_informacao1,
                  text ="\n                In the Program separator Insert | Run", font="verdana 9")
label_mensagem5_MNIST = Label(frame_menu_informacao1,
                  text ="\n                Insert the parametres and after edit|run", font="verdana 9 bold")
label_mensagem6_MNIST = Label(frame_menu_informacao1,
                  text ="\n\n              2 - Results, choose the results for each Arquitecture", font="verdana 10 bold")
label_mensagem7_MNIST = Label(frame_menu_informacao1,
                  text ="\n                Show PLot - See  each/all the results in a plot ", font="verdana 9")
label_mensagem8_MNIST = Label(frame_menu_informacao1,
                  text ="                  Print each result in Iphyton", font="verdana 9")


label_mensagem3_MNIST.grid(row=4, column=0)
label_mensagem4_MNIST.grid(row=5, column=0)
label_mensagem5_MNIST.grid(row=6, column=0)
label_mensagem6_MNIST.grid(row=7, column=0)
label_mensagem7_MNIST.grid(row=8, column=0)
label_mensagem8_MNIST.grid(row=9, column=0)


#----------------------------------- Widgets Escolhe se quer modelo pré defenido ou editar --------------------

label_ins1_insere3 = Label(frame_menu_inserir3, width=33, height=2,
                      text ="Check the progress in the Console.", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins_insere3 = Label(frame_menu_inserir3, width=33, height=3,
                      text ="This process can take several hours.", font=('verdana', 10), justify=CENTER)
label_ins1_insere3.grid(row=0, column=0)
label_ins_insere3.grid(row=1, column=0)


#----------------------------------- Widgets Escolhe se quer modelo 1d pré defenido ou editar --------------------

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
                                 font=('verdana', 9, 'bold'),command=runmodel_alexnet_2d)
cmd_editamodelo.grid(row=2,column=1)



#----------------------------------- Widgets Escolhe se quer modelo 2d pré defenido ou editar --------------------

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
                                 font=('verdana', 9, 'bold'),command=aux1_insere_convolucao)
cmd_escolhemodelo1_edit.grid(row=4,column=1)

#----------------------------------- Widgets Inserir dados para rede para 1 dimensão------------------------------------

label_ins = Label(frame_menu_inserir1, width=33, height=2,
                  text ="Insert Parameters for 1 dimension", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins0 = Label(frame_menu_inserir1, width=8, height=2,
                  text ="Values:", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins01 = Label(frame_menu_inserir1, width=15, height=1,
                  text ="Recomended:", font=('verdana', 8, 'bold'), justify=CENTER)
label_ins1 = Label(frame_menu_inserir1, width=33, height=1,
                  text ="Signals sampling frequency: ", font="verdana 10")
label_ins1_1 = Label(frame_menu_inserir1, width=10, height=1,
                  text ="200", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins2 = Label(frame_menu_inserir1, width=33, height=1,
                  text ="Times should the algorithm run:", font="verdana 10")
label_ins2_1 = Label(frame_menu_inserir1, width=10, height=1,
                  text ="20", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins4 = Label(frame_menu_inserir1, width=33, height=1,
                  text ="Epochs update relevant:", font="verdana 10")
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
                  text ="Value for the early stopping:", font="verdana 10")
label_ins7_1 = Label(frame_menu_inserir1, width=10, height=1,
                  text ="40", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins11 = Label(frame_menu_inserir1, width=33, height=1,
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
label_ins7_1.grid(row=7, column=2)
label_ins11.grid(row=8, column=0)

texto_ins1.grid(row=1,column=1)
texto_ins2.grid(row=2,column=1)
texto_ins4.grid(row=4,column=1)
texto_ins5.grid(row=5,column=1)
texto_ins6.grid(row=6,column=1)
texto_ins7.grid(row=7,column=1)

cmd_verificavalores = Button(frame_menu_inserir1, text="Insert/Save",
                             font=('verdana', 9, 'bold'),command=Insere)
cmd_verificavalores.grid(row=11,column=1)

#----------------------------------- Widgets Inserir dados para rede para 2 dimensões------------------------------------

label_ins2d = Label(frame_menu_inserir12d, width=33, height=2,
                  text ="Insert Parameters for 2 dimensions", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins02d = Label(frame_menu_inserir12d, width=8, height=2,
                  text ="Values:", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins012d = Label(frame_menu_inserir12d, width=15, height=1,
                  text ="Recomended:", font=('verdana', 8, 'bold'), justify=CENTER)
label_ins22d = Label(frame_menu_inserir12d, width=33, height=1,
                  text ="Initial learning rate:", font="verdana 10")
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
                  text ="Value for the early stopping:", font="verdana 10")
label_ins6_12d = Label(frame_menu_inserir12d, width=10, height=1,
                  text ="10", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins72d = Label(frame_menu_inserir12d, width=33, height=1,
                  text ="Monte Carlo Value:", font="verdana 10")
label_ins7_12d = Label(frame_menu_inserir12d, width=10, height=1,
                  text ="50", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins02d1 = Label(frame_menu_inserir12d, width=33, height=1,
                  text ="Image Shape - width:", font="verdana 10")
label_ins7_12d1 = Label(frame_menu_inserir12d, width=10, height=1,
                  text ="28", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins02d2 = Label(frame_menu_inserir12d, width=33, height=1,
                  text ="Image Shape - height:", font="verdana 10")
label_ins7_12d2 = Label(frame_menu_inserir12d, width=10, height=1,
                  text ="28", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins02d3 = Label(frame_menu_inserir12d, width=33, height=1,
                  text ="Image Shape - channels:", font="verdana 10")
label_ins7_12d3 = Label(frame_menu_inserir12d, width=10, height=1,
                  text ="1", font=('verdana', 10, 'bold'), justify=CENTER)
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
label_ins7_12d1.grid(row=8, column=2)
label_ins7_12d2.grid(row=9, column=2)
label_ins7_12d3.grid(row=10, column=2)

texto_ins22d.grid(row=2,column=1)
texto_ins42d.grid(row=4,column=1)
texto_ins52d.grid(row=5,column=1)
texto_ins62d.grid(row=6,column=1)
texto_ins72d.grid(row=7,column=1)
texto_ins82d.grid(row=8,column=1)
texto_ins92d.grid(row=9,column=1)
texto_ins02d.grid(row=10,column=1)
label_ins112d.grid(row=11,column=1)

cmd_verificavalores2d = Button(frame_menu_inserir12d, text="Insert/Save",
                             font=('verdana', 9, 'bold'),command=Insere2d)
cmd_verificavalores2d.grid(row=12,column=1)

#----------------------------------- Widgets Inserir camada de dense------------------------------------

label_ins2ddense = Label(frame_menu_dense, width=33, height=2,
                  text ="Insert dense layer", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins2ddense1 = Label(frame_menu_dense, width=15, height=2,
                  text ="Values", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins02ddense = Label(frame_menu_dense, width=33, height=2,
                  text ="Number of neurons:", font=('verdana', 10), justify=CENTER)
label_ins2ddense = Label(frame_menu_dense, width=33, height=2,
                  text ="Insert dense layer", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins2ddenseeb = Label(frame_menu_dense, width=33, height=2,
                  text ="        ", font=('verdana', 10, 'bold'), justify=CENTER)

label_ins2ddense1d = Label(frame_menu_dense, width=33, height=2,
                  text ="Insert another layer", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins2ddenseeb2 = Label(frame_menu_dense, width=33, height=2,
                  text ="        ", font=('verdana', 10, 'bold'), justify=CENTER)


label_ins2ddense.grid(row=0, column=0)
label_ins2ddense1.grid(row=0, column=1)
label_ins02ddense.grid(row=1, column=0)
label_ins2ddenseeb.grid(row=3, column=0)
label_ins2ddense1d.grid(row=4, column=0)
label_ins2ddenseeb2.grid(row=7, column=0)


texto_number_of_neurons = Entry(frame_menu_dense, width=8, font="verdana 9", justify=CENTER)
texto_number_of_neurons.grid(row=1,column=1)

cmd_verificavalores2ddense = Button(frame_menu_dense, text="Insert",
                             font=('verdana', 9, 'bold'),command=aux1_insere_dense)
cmd_verificavalores2ddense.grid(row=2,column=1)

cmd_verificavalores2dpooling = Button(frame_menu_dense, text="Convolution",
                             font=('verdana', 9, 'bold'),command=aux1_insere_convolucao)
cmd_verificavalores2dpooling.grid(row=6,column=1)
cmd_verificavalores2dddense = Button(frame_menu_dense, text="Pooling",
                             font=('verdana', 9, 'bold'),command=aux1_insere_pooling)
cmd_verificavalores2dddense.grid(row=6,column=0)

cmd_verificavalores2dddense = Button(frame_menu_dense, text="Dense",
                             font=('verdana', 9, 'bold'),command=aux1_insere_dense)
cmd_verificavalores2dddense.grid(row=10,column=0)
cmd_verificavalores2dcddense = Button(frame_menu_dense, text="Run Model",
                             font=('verdana', 9, 'bold'),command=aux1_insere_convolucao)
cmd_verificavalores2dcddense.grid(row=10,column=1)


#----------------------------------- Widgets Inserir camada de pooling------------------------------------

label_ins2dpooling = Label(frame_menu_pooling, width=33, height=2,
                  text ="Insert pooling layer", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins02dpooling = Label(frame_menu_pooling, width=33, height=2,
                  text ="Number of kernels:", font=('verdana', 10), justify=CENTER)
label_ins22dpooling = Label(frame_menu_pooling, width=33, height=1,
                  text ="Number os strides:", font="verdana 10")
label_ins2dpooling1 = Label(frame_menu_pooling, width=15, height=2,
                  text ="Values", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins2dpooling1c = Label(frame_menu_pooling, width=15, height=2,
                  text ="     ", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins2dpooling1c1 = Label(frame_menu_pooling, width=15, height=2,
                  text ="     ", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins2dpooling1c2 = Label(frame_menu_pooling, width=15, height=2,
                  text ="     ", font=('verdana', 10, 'bold'), justify=CENTER)

label_ins2dpooling1e = Label(frame_menu_pooling, width=33, height=2,
                  text ="Insert another layer", font=('verdana', 10, 'bold'), justify=CENTER)


label_ins2dpooling1.grid(row=0, column=1)
label_ins2dpooling.grid(row=0, column=0)
label_ins02dpooling.grid(row=1, column=0)
label_ins22dpooling.grid(row=2, column=0)
label_ins2dpooling1c.grid(row=4, column=0)
label_ins2dpooling1c1.grid(row=6, column=0)
label_ins2dpooling1e.grid(row=5, column=0)
label_ins2dpooling1c2.grid(row=7, column=0)


texto_number_of_kernels1 = Entry(frame_menu_pooling, width=8, font="verdana 9", justify=CENTER)
texto_number_of_kernels1.grid(row=1,column=1)
texto_number_of_strides = Entry(frame_menu_pooling, width=8, font="verdana 9", justify=CENTER)
texto_number_of_strides.grid(row=2,column=1)


cmd_verificavalores2dpooling = Button(frame_menu_pooling, text="Insert",
                             font=('verdana', 9, 'bold'),command=aux1_insere_pooling)
cmd_verificavalores2dpooling.grid(row=3,column=1)

cmd_verificavalores2dpooling = Button(frame_menu_pooling, text="Convolution",
                             font=('verdana', 9, 'bold'),command=aux1_insere_convolucao)
cmd_verificavalores2dpooling.grid(row=6,column=1)
cmd_verificavalores2dpooling = Button(frame_menu_pooling, text="Pooling",
                             font=('verdana', 9, 'bold'),command=aux1_insere_pooling)
cmd_verificavalores2dpooling.grid(row=6,column=0)

cmd_verificavalores2dpooling = Button(frame_menu_pooling, text="Dense",
                             font=('verdana', 9, 'bold'),command=aux1_insere_dense)
cmd_verificavalores2dpooling.grid(row=10,column=0)
cmd_verificavalores2dcpooling = Button(frame_menu_pooling, text="Run Model",
                             font=('verdana', 9, 'bold'),command=aux1_insere_convolucao)
cmd_verificavalores2dcpooling.grid(row=10,column=1)


#----------------------------------- Widgets Inserir camada de convolucao------------------------------------

label_ins2dconvolucao = Label(frame_menu_convolucao, width=33, height=2,
                  text ="Insert convolution Layer", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins2dconvolucaov = Label(frame_menu_convolucao, width=15, height=2,
                  text ="Values", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins02dconvolucao = Label(frame_menu_convolucao, width=33, height=1,
                  text ="Neurons (Filters):", font=('verdana', 9), justify=CENTER)
label_ins22dconvolucao = Label(frame_menu_convolucao, width=33, height=1,
                  text ="Kernel size:", font="verdana 10")
label_ins42dconvolucao = Label(frame_menu_convolucao, width=33, height=1,
                  text ="Number os strides:", font="verdana 10")
label_ins52dconvolucao = Label(frame_menu_convolucao, width=33, height=1,
                  text ="Padding (Same=0 | Valid=1):", font="verdana 10")
label_ins52dconvolucaoe1 = Label(frame_menu_convolucao, width=33, height=1,
                  text ="                   ", font="verdana 10")
label_ins2dconvolucao1e = Label(frame_menu_convolucao, width=33, height=2,
                  text ="Insert another layer", font=('verdana', 10, 'bold'), justify=CENTER)
label_ins52dconvolucaoe13 = Label(frame_menu_convolucao, width=33, height=1,
                  text ="                   ", font="verdana 10")

label_ins2dconvolucaov.grid(row=0, column=1)
label_ins2dconvolucao.grid(row=0, column=0)
label_ins02dconvolucao.grid(row=1, column=0)
label_ins22dconvolucao.grid(row=2, column=0)
label_ins42dconvolucao.grid(row=3, column=0)
label_ins52dconvolucao.grid(row=4, column=0)
label_ins52dconvolucaoe1.grid(row=5, column=0)
label_ins2dconvolucao1e.grid(row=7, column=0)
label_ins52dconvolucaoe13.grid(row=9, column=0)

texto_number_of_neurons2 = Entry(frame_menu_convolucao, width=8, font="verdana 9", justify=CENTER)
texto_number_of_neurons2.grid(row=1,column=1)
texto_kernel_size = Entry(frame_menu_convolucao, width=8, font="verdana 9", justify=CENTER)
texto_kernel_size.grid(row=2,column=1)
texto_strides = Entry(frame_menu_convolucao, width=8, font="verdana 9", justify=CENTER)
texto_strides.grid(row=3,column=1)
texto_padding = Entry(frame_menu_convolucao, width=8, font="verdana 9", justify=CENTER)
texto_padding.grid(row=4,column=1)



cmd_verificavalores2dconvolucao = Button(frame_menu_convolucao, text="Insert",
                             font=('verdana', 9, 'bold'),command=aux1_insere_convolucao)
cmd_verificavalores2dconvolucao.grid(row=6,column=1)
cmd_verificavalores2dconvolucao = Button(frame_menu_convolucao, text="Convolution",
                             font=('verdana', 9, 'bold'),command=aux1_insere_convolucao)
cmd_verificavalores2dconvolucao.grid(row=8,column=1)
cmd_verificavalores2dconvolucao = Button(frame_menu_convolucao, text="Pooling",
                             font=('verdana', 9, 'bold'),command=aux1_insere_pooling)
cmd_verificavalores2dconvolucao.grid(row=8,column=0)

cmd_verificavalores2dconvolucao = Button(frame_menu_convolucao, text="Dense",
                             font=('verdana', 9, 'bold'),command=aux1_insere_dense)
cmd_verificavalores2dconvolucao.grid(row=10,column=0)
cmd_verificavalores2dconvolucao = Button(frame_menu_convolucao, text="Run Model",
                             font=('verdana', 9, 'bold'),command=aux1_insere_convolucao)
cmd_verificavalores2dconvolucao.grid(row=10,column=1)


#------------------------------------------------------------ 
#Cancela o redimensionamento
menu_inicial.resizable(False, False)
menu_inicial.mainloop()


