import sys
import os
# importing the library to save the results
import pickle
# importing the library to check the dataset files
from os import path
import csv
import matplotlib.pyplot as plt
import os
import warnings
import numpy as np


# ----------------------- Print 1d LeNet   ---------------------

def lenet_1d_all_plot(result = "default", classes = 5):
        classes = int(classes)
        with open('Results\LeNet1d\AccLenet1.txt', 'rb') as handle:
            a = pickle.load(handle)
        x = np.array(range(0, classes))
        y = np.zeros(classes)
        for i in range (len(a)):
          y[i]=a[i]          
        with open('Results\LeNet1d\AucLeNet1.txt', 'rb') as handle1:
            b = pickle.load(handle1)
        x = np.array(range(0, classes))
        y1 = np.zeros(classes)
        for i in range (len(b)):
          y1[i]=b[i]          
        with open(rb'Results\LeNet1d\NpvLeNet1.txt', 'rb') as handle1:
            c = pickle.load(handle1)
        x = np.array(range(0, classes))
        y2 = np.zeros(classes)
        for i in range (len(b)):
          y2[i]=c[i]           
        with open('Results\LeNet1d\PpvLeNet1.txt', 'rb') as handle1:
            d = pickle.load(handle1)
        x = np.array(range(0, classes))
        y3 = np.zeros(classes)
        for i in range (len(b)):
          y3[i]=d[i]             
        with open('Results\LeNet1d\SenLeNet1.txt', 'rb') as handle1:
            e = pickle.load(handle1)
        x = np.array(range(0, classes))
        y4 = np.zeros(classes)
        for i in range (len(b)):
          y4[i]=e[i]           
        with open('Results\LeNet1d\SpeLeNet1.txt', 'rb') as handle1:
            f = pickle.load(handle1)
        x = np.array(range(0, classes))
        y5 = np.zeros(classes)
        for i in range (len(b)):
          y5[i]=f[i]    
        plt.title("LeNet - ACC | ACU | NPV | PPV | SPE | SEN")
        plt.xlabel("Classes")
        plt.ylabel("ACC | ACU | NPV | PPV | SPE | SEN")
        plt.plot(x, y, color = "Orange", marker = "o", label = "Acc values")
        plt.plot(x, y1, color = "Green", marker = "o", label = "Auc values")
        plt.plot(x, y2, color = "blue", marker = "o", label = "NPV values")
        plt.plot(x, y3, color = "red", marker = "o", label = "PPV values")
        plt.plot(x, y4, color = "black", marker = "o", label = "Sen values")
        plt.plot(x, y5, color = "Purple", marker = "o", label = "Spe values")
        plt.grid()
        plt.legend()
        plt.show()    

def lenet_1d_one_plot(result = "default", classes = 5):
    classes = int(classes)
    if result == "Acc-LeNet":
        with open('Results\LeNet1d\AccLenet1.txt', 'rb') as handle:
            a = pickle.load(handle)
        x = np.array(range(0, classes))
        y = np.zeros(classes)
        for i in range (len(a)):
          y[i]=a[i]    
        plt.title("LeNet - Acc values")
        plt.xlabel("Classes")
        plt.ylabel("Acc values")
        plt.plot(x, y, color = "Orange", marker = "o", label = "Acc values")
        plt.grid()
        plt.legend()
        plt.show()
              
    elif result == "Auc-LeNet":
        with open('Results\LeNet1d\AucLenet1.txt', 'rb') as handle1:
            b = pickle.load(handle1)
        x = np.array(range(0, classes))
        y1 = np.zeros(classes)
        for i in range (len(b)):
          y1[i]=b[i] 
        plt.title("LeNet - Auc values")
        plt.xlabel("Classes")
        plt.ylabel("Auc values")
        plt.plot(x, y1, color = "Green", marker = "o", label = "Auc values")
        plt.grid()
        plt.legend()
        plt.show()
        
    elif result == "Npv-LeNet":
        with open(rb'Results\LeNet1d\NpvLenet1.txt', 'rb') as handle1:
            c = pickle.load(handle1)
        x = np.array(range(0, classes))
        y2 = np.zeros(classes)
        for i in range (len(c)):
          y2[i]=c[i]      
        plt.title("LeNet - NPV values")
        plt.xlabel("Classes")
        plt.ylabel("NPV values")
        plt.plot(x, y2, color = "blue", marker = "o", label = "NPV values")
        plt.grid()
        plt.legend()
        plt.show()    
        
    elif result == "Ppv-LeNet":
        with open('Results\LeNet1d\PpvLenet1.txt', 'rb') as handle1:
            d = pickle.load(handle1)
        x = np.array(range(0, classes))
        y3 = np.zeros(classes)
        for i in range (len(d)):
          y3[i]=d[i]   
        plt.title("LeNet - PPV values")
        plt.xlabel("Classes")
        plt.ylabel("PPV values")
        plt.plot(x, y3, color = "red", marker = "o", label = "PPV values")
        plt.grid()
        plt.legend()
        plt.show()    
        
    elif result == "Sen-LeNet":
        with open('Results\LeNet1d\SenLenet1.txt', 'rb') as handle1:
            e = pickle.load(handle1)
        x = np.array(range(0, classes))
        y4 = np.zeros(classes)
        for i in range (len(e)):
          y4[i]=e[i]      
        plt.title("LeNet - Sen values")
        plt.xlabel("Classes")
        plt.ylabel("Sen values")
        plt.plot(x, y4, color = "black", marker = "o", label = "Sen values")
        plt.grid()
        plt.legend()
        plt.show()

    elif result == "Spe-LeNet":
        with open('Results\LeNet1d\SpeLenet1.txt', 'rb') as handle1:
            f = pickle.load(handle1)
        x = np.array(range(0, classes))
        y5 = np.zeros(classes)
        for i in range (len(f)):
          y5[i]=f[i]    
        plt.xlabel("Classes")
        plt.ylabel("Spe values")
        plt.plot(x, y5, color = "Purple", marker = "o", label = "Spe values")
        plt.grid()
        plt.legend()
        plt.show()

def lenet_1d_print(result ="default", classes = 5):
    if result == "Acc-LeNet":
        object_Acc_Generic = []
        with (open("Results\LeNet1d\AccLeNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_Acc_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_Acc_Generic)   
                 
    elif result == "Auc-LeNet":
        object_Auc_Generic = []
        with (open("Results\LeNet1d\AucLeNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_Auc_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_Auc_Generic)
                 
    elif result == "Npv-LeNet":
        object_NPV_Generic = []
        with (open(r"Results\LeNet1d\NpvLeNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_NPV_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_NPV_Generic)
                 
    elif result == "Ppv-LeNet":
        object_PPV_Generic = []
        with (open("Results\LeNet1d\PpvLeNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_PPV_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_PPV_Generic)
                 
    elif result == "Sen-LeNet":
        object_SEN_Generic = []
        with (open("Results\LeNet1d\SenLeNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_SEN_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_SEN_Generic)

    elif result == "Spe-LeNet":
        object_SPE_Generic = []
        with (open("Results\LeNet1d\SpeLeNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_SPE_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_SPE_Generic)
    else:    
            print("Select the result in the select box") 
            



#------------------------ Print 1d Alex Net -------------------

def alexnet_1d_all_plot(result = "default", classes = 5):
        classes = int(classes)
        print(classes)
        classes = int(classes)
        with open('Results\AlexNet1d\AccAlexNet1.txt', 'rb') as handle:
            a = pickle.load(handle)
        x = np.array(range(0, classes))
        y = np.zeros(classes)
        for i in range (len(a)):
          y[i]=a[i]          
        with open('Results\AlexNet1d\AucAlexNet1.txt', 'rb') as handle1:
            b = pickle.load(handle1)
        x = np.array(range(0, classes))
        y1 = np.zeros(classes)
        for i in range (len(b)):
          y1[i]=b[i]          
        with open(rb'Results\AlexNet1d\NpvAlexNet1.txt', 'rb') as handle1:
            c = pickle.load(handle1)
        x = np.array(range(0, classes))
        y2 = np.zeros(classes)
        for i in range (len(b)):
          y2[i]=c[i]           
        with open('Results\AlexNet1d\PpvAlexNet1.txt', 'rb') as handle1:
            d = pickle.load(handle1)
        x = np.array(range(0, classes))
        y3 = np.zeros(classes)
        for i in range (len(b)):
          y3[i]=d[i]             
        with open('Results\AlexNet1d\SenAlexNet1.txt', 'rb') as handle1:
            e = pickle.load(handle1)
        x = np.array(range(0, classes))
        y4 = np.zeros(classes)
        for i in range (len(b)):
          y4[i]=e[i]           
        with open('Results\AlexNet1d\SenAlexNet1.txt', 'rb') as handle1:
            f = pickle.load(handle1)
        x = np.array(range(0, classes))
        y5 = np.zeros(classes)
        for i in range (len(b)):
          y4[i]=f[i]    
        plt.title("AlexNet ACC | ACU | NPV | PPV | SPE | SEN")
        plt.xlabel("Classes")
        plt.ylabel("ACC | ACU | NPV | PPV | SPE | SEN")
        plt.plot(x, y, color = "Orange", marker = "o", label = "Acc values")
        plt.plot(x, y1, color = "Green", marker = "o", label = "Auc values")
        plt.plot(x, y2, color = "blue", marker = "o", label = "NPV values")
        plt.plot(x, y3, color = "red", marker = "o", label = "PPV values")
        plt.plot(x, y4, color = "black", marker = "o", label = "Sen values")
        plt.plot(x, y5, color = "Purple", marker = "o", label = "Spe values")
        plt.grid()
        plt.legend()
        plt.show()


def alexnet_1d_one_plot(result = "default", classes = 5):
    classes = int(classes)
    if result == "Acc-AlexNet":
        with open('Results\AlexNet1d\AccAlexNet1.txt', 'rb') as handle:
            a = pickle.load(handle)
        x = np.array(range(0, classes))
        y = np.zeros(classes)
        for i in range (len(a)):
          y[i]=a[i]   
        plt.title("Acc values")
        plt.xlabel("Classes")
        plt.ylabel("Acc values")
        plt.plot(x, y, color = "Orange", marker = "o", label = "Acc values")
        plt.grid()
        plt.legend()
        plt.show()
        
    elif result == "Auc-AlexNet":
        with open('Results\AlexNet1d\AucAlexNet1.txt', 'rb') as handle1:
            b = pickle.load(handle1)
        x = np.array(range(0, classes))
        y1 = np.zeros(classes)
        for i in range (len(b)):
          y1[i]=b[i]  
        plt.title("AlexNet - Auc values")
        plt.xlabel("Classes")
        plt.ylabel("Auc values")
        plt.plot(x, y1, color = "Green", marker = "o", label = "Auc values")
        plt.grid()    
        plt.legend()
        plt.show()

    
    elif result == "Npv-AlexNet":
        with open(rb'Results\AlexNet1d\NpvAlexNet1.txt', 'rb') as handle1:
            c = pickle.load(handle1)
        x = np.array(range(0, classes))
        y2 = np.zeros(classes)
        for i in range (len(c)):
          y2[i]=c[i]        
        plt.title("NPV values")
        plt.xlabel("Classes")
        plt.ylabel("NPV values")
        plt.plot(x, y2, color = "blue", marker = "o", label = "NPV values")
        plt.grid()        
        plt.legend()
        plt.show()
        
    elif result == "Ppv-AlexNet":
        with open('Results\AlexNet1d\PpvAlexNet1.txt', 'rb') as handle1:
            d = pickle.load(handle1)
        x = np.array(range(0, classes))
        y3 = np.zeros(classes)
        for i in range (len(d)):
          y3[i]=d[i]    
        plt.title("AlexNet - PPV valuess")
        plt.xlabel("Classes")
        plt.ylabel("PPV values")
        plt.plot(x, y3, color = "red", marker = "o", label = "PPV values")
        plt.grid()
        plt.legend()
        plt.show()
      
    elif result == "Sen-AlexNet":
        with open('Results\AlexNet1d\SenAlexNet1.txt', 'rb') as handle1:
            e = pickle.load(handle1)
        x = np.array(range(0, classes))
        y4 = np.zeros(10)
        for i in range (len(e)):
          y4[i]=e[i]     
        plt.title("AlexNet - Sen")
        plt.xlabel("Classes")
        plt.ylabel("Sen values")
        plt.plot(x, y4, color = "black", marker = "o", label = "Sen values")
        plt.grid()
        plt.legend()
        plt.show()

    elif result == "Spe-AlexNet":
        with open('Results\AlexNet1d\SenAlexNet1.txt', 'rb') as handle1:
            f = pickle.load(handle1)
        x = np.array(range(0, classes))
        y5 = np.zeros(classes)
        for i in range (len(f)):
          y5[i]=f[i]  
        plt.title("AlexNet - Spe")
        plt.xlabel("Classes")
        plt.ylabel("Spe values")
        plt.plot(x, y5, color = "Purple", marker = "o", label = "Spe values")
        plt.grid()
        plt.legend()
        plt.show()

def alexnet_1d_print(result ="default", classes = 5):
    
    if result == "Acc-AlexNet":
        object_Acc_alexnet = []
        with (open("Results\AlexNet1d\AccAlexNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_Acc_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_Acc_alexnet)   
                 
    elif result == "Auc-AlexNet":
        object_Auc_alexnet = []
        with (open("Results\AlexNet1d\AucAlexNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_Auc_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_Auc_alexnet)
                 
    elif result == "Npv-AlexNet":
        object_NPV_alexnet = []
        with (open(rb"Results\AlexNet1d\NpvAlexNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_NPV_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_NPV_alexnet)
                 
    elif result == "Ppv-AlexNet":
        object_PPV_alexnet = []
        with (open("Results\AlexNet1d\PpvAlexNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_PPV_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_PPV_alexnet)
                 
    elif result == "Sen-AlexNet":
        object_SEN_alexnet = []
        with (open("Results\AlexNet1d\SenAlexNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_SEN_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_SEN_alexnet)

    elif result == "Spe-AlexNet":
        object_SPE_alexnet = []
        with (open("Results\AlexNet1d\SpeAlexNet1.txt","rb")) as openfile:
              while True:
                 try:
                    object_SPE_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_SPE_alexnet)

    else:    
            print("Select the result in the select box") 
            
            
        
            
# ---------------------  Print 1d Edit    ---------------


def edit_1d_all_plot(result = "default", classes = 5):
        classes = int(classes)
        with open('Results\Edit1d\AccEdit1d.txt', 'rb') as handle:
            a = pickle.load(handle)
        x = np.array(range(0, classes))
        y = np.zeros(classes)
        for i in range (len(a)):
          y[i]=a[i] 
        with open('Results\Edit1d\AucEdit1d.txt', 'rb') as handle1:
            b = pickle.load(handle1)
        x = np.array(range(0, classes))
        y1 = np.zeros(classes)
        for i in range (len(b)):
          y1[i]=b[i]         
        with open(rb'Results\Edit1d\PpvEdit1d.txt', 'rb') as handle2:
            c = pickle.load(handle2)
        x = np.array(range(0, classes))
        y2 = np.zeros(classes)
        for i in range (len(c)):
          y2[i]=c[i]      
        with open(r'Results\Edit1d\NpvEdit1d.txt', 'rb') as handle3:
            d = pickle.load(handle3)
        x = np.array(range(0, classes))
        y3 = np.zeros(classes)
        for i in range (len(d)):
          y3[i]=d[i]     
        with open('Results\Edit1d\SenEdit1d.txt', 'rb') as handle4:
            e = pickle.load(handle4)
        x = np.array(range(0, classes))
        y4 = np.zeros(classes)
        for i in range (len(e)):
          y4[i]=e[i] 
        with open('Results\Edit1d\SpeEdit1d.txt', 'rb') as handle5:
            f = pickle.load(handle5)
        x = np.array(range(0, classes))
        y5 = np.zeros(classes)
        for i in range (len(f)):
          y5[i]=f[i]    
        plt.title("Edit Results - ACC | ACU | NPV | PPV | SPE | SEN")
        plt.xlabel("Classes")
        plt.ylabel("ACC | ACU | NPV | PPV | SPE | SEN")
        plt.plot(x, y, color = "Orange", marker = "o", label = "Acc values")
        plt.plot(x, y1, color = "Green", marker = "o", label = "Auc values")
        plt.plot(x, y2, color = "blue", marker = "o", label = "NPV values")
        plt.plot(x, y3, color = "red", marker = "o", label = "PPV values")
        plt.plot(x, y4, color = "black", marker = "o", label = "Sen values")
        plt.plot(x, y5, color = "Purple", marker = "o", label = "Spe values")
        plt.grid()
        plt.legend()
        plt.show()

def edit_1d_one_plot(result = "default", classes = 5):
    classes = int(classes)
    if result == "Acc-Edit1d":
        with open('Results\Edit1d\AccEdit1d.txt', 'rb') as handle:
            a = pickle.load(handle)
        x = np.array(range(0, classes))
        y = np.zeros(classes)
        for i in range (len(a)):
          y[i]=a[i] 
        plt.title("LeNet - Acc values")
        plt.xlabel("Classes")
        plt.ylabel("Acc values")
        plt.plot(x, y, color = "Orange", marker = "o", label = "Acc values")
        plt.grid()
        plt.legend()
        plt.show()
    
    elif result == "Auc-Edit1d":
        with open('Results\Edit1d\AucEdit1d.txt', 'rb') as handle1:
            b = pickle.load(handle1)
        x = np.array(range(0, classes))
        y1 = np.zeros(classes)
        for i in range (len(b)):
          y1[i]=b[i]      
        plt.title("LeNet - Auc values")
        plt.xlabel("Classes")
        plt.ylabel("Auc values")
        plt.plot(x, y1, color = "Green", marker = "o", label = "Auc values")
        plt.grid()
        plt.legend()
        plt.show()
        
    elif result == "Npv-Edit1d":
        with open(rb'Results\Edit1d\NpvEdit1d.txt', 'rb') as handle2:
            c = pickle.load(handle2)
        x = np.array(range(0, classes))
        y2 = np.zeros(classes)
        for i in range (len(c)):
          y2[i]=c[i]  
        plt.title("LeNet - NPV values")
        plt.xlabel("Classes")
        plt.ylabel("NPV values")
        plt.plot(x, y2, color = "blue", marker = "o", label = "NPV values")
        plt.grid()
        plt.legend()
        plt.show()    
        
    elif result == "Ppv-Edit1d":
        with open('Results\Edit1d\PpvEdit1d.txt', 'rb') as handle3:
            d = pickle.load(handle3)
        x = np.array(range(0, classes))
        y3 = np.zeros(classes)
        for i in range (len(d)):
          y3[i]=d[i]    
        plt.title("LeNet - PPV values")
        plt.xlabel("Classes")
        plt.ylabel("PPV values")
        plt.plot(x, y3, color = "red", marker = "o", label = "PPV values")
        plt.grid()
        plt.legend()
        plt.show()    
        
    elif result == "Sen-Edit1d":
        with open('Results\Edit1d\SenEdit1d.txt', 'rb') as handle4:
            e = pickle.load(handle4)
        x = np.array(range(0, classes))
        y4 = np.zeros(classes)
        for i in range (len(e)):
          y4[i]=e[i] 
        plt.title("LeNet - Sen values")
        plt.xlabel("Classes")
        plt.ylabel("Sen values")
        plt.plot(x, y4, color = "black", marker = "o", label = "Sen values")
        plt.grid()
        plt.legend()
        plt.show()

    elif result == "Spe-Edit1d":
        with open('Results\Edit1d\SpeEdit1d.txt', 'rb') as handle5:
            f = pickle.load(handle5)
        x = np.array(range(0, classes))
        y5 = np.zeros(classes)
        for i in range (len(f)):
          y5[i]=f[i]    
        plt.title("LeNet - Spe values")
        plt.xlabel("Classes")
        plt.ylabel("Spe values")
        plt.plot(x, y5, color = "Purple", marker = "o", label = "Spe values")
        plt.grid()
        plt.legend()
        plt.show()

def edit_1d_print(result ="default", classes = 5):
    
    
    if result == "Acc-Edit1d":
        object_Acc_alexnet = []
        with (open("Results\AlexEdit1d\AccEdit1d.txt","rb")) as openfile:
              while True:
                 try:
                    object_Acc_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_Acc_alexnet)   
                 
    elif result == "Auc-Edit1d":
        object_Auc_alexnet = []
        with (open("Results\AlexEdit1d\AucEdit1d.txt","rb")) as openfile:
              while True:
                 try:
                    object_Auc_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_Auc_alexnet)
                 
    elif result == "Npv-Edit1d":
        object_NPV_alexnet = []
        with (open(rb"Results\AlexEdit1d\NpvEdit1d.txt","rb")) as openfile:
              while True:
                 try:
                    object_NPV_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_NPV_alexnet)
                 
    elif result == "Ppv-Edit1d":
        object_PPV_alexnet = []
        with (open("Results\AlexEdit1d\PpvEdit1d.txt","rb")) as openfile:
              while True:
                 try:
                    object_PPV_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_PPV_alexnet)
                 
    elif result == "Sen-Edit1d":
        object_SEN_alexnet = []
        with (open("Results\AlexEdit1d\SenEdit1d.txt","rb")) as openfile:
              while True:
                 try:
                    object_SEN_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_SEN_alexnet)

    elif result == "Spe-Edit1d":
        object_SPE_alexnet = []
        with (open("Results\AlexEdit1d\SpeEdit1d.txt","rb")) as openfile:
              while True:
                 try:
                    object_SPE_alexnet.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_SPE_alexnet)

    else:    
            print("Select the result in the select box")
 
    
 
# ----------------Print 2d AlexNet ---------------


def alexnet_2d_all_plot(result = "default", classes = 5):
        classes = int(classes)
        print(classes)
        classes = int(classes)
        with open('Results\AlexNet2d\AccAlexNet2.txt', 'rb') as handle:
            a = pickle.load(handle)
        x = np.array(range(0, classes))
        y = np.zeros(classes)
        for i in range (len(a)):
          y[i]=a[i]          
        with open('Results\AlexNet2d\AucAlexNet2.txt', 'rb') as handle1:
            b = pickle.load(handle1)
        x = np.array(range(0, classes))
        y1 = np.zeros(classes)
        for i in range (len(b)):
          y1[i]=b[i]          
        with open(rb'Results\AlexNet2d\NpvAlexNet2.txt', 'rb') as handle1:
            c = pickle.load(handle1)
        x = np.array(range(0, classes))
        y2 = np.zeros(classes)
        for i in range (len(b)):
          y2[i]=c[i]           
        with open('Results\AlexNet2d\PpvAlexNet2.txt', 'rb') as handle1:
            d = pickle.load(handle1)
        x = np.array(range(0, classes))
        y3 = np.zeros(classes)
        for i in range (len(b)):
          y3[i]=d[i]             
        with open('Results\AlexNet2d\SenAlexNet2.txt', 'rb') as handle1:
            e = pickle.load(handle1)
        x = np.array(range(0, classes))
        y4 = np.zeros(classes)
        for i in range (len(b)):
          y4[i]=e[i]           
        with open('Results\AlexNet2d\SenAlexNet2.txt', 'rb') as handle1:
            f = pickle.load(handle1)
        x = np.array(range(0, classes))
        y5 = np.zeros(classes)
        for i in range (len(b)):
          y4[i]=f[i]    
        plt.title("AlexNet ACC | ACU | NPV | PPV | SPE | SEN")
        plt.xlabel("Classes")
        plt.ylabel("ACC | ACU | NPV | PPV | SPE | SEN")
        plt.plot(x, y, color = "Orange", marker = "o", label = "Acc values")
        plt.plot(x, y1, color = "Green", marker = "o", label = "Auc values")
        plt.plot(x, y2, color = "blue", marker = "o", label = "NPV values")
        plt.plot(x, y3, color = "red", marker = "o", label = "PPV values")
        plt.plot(x, y4, color = "black", marker = "o", label = "Sen values")
        plt.plot(x, y5, color = "Purple", marker = "o", label = "Spe values")
        plt.grid()
        plt.legend()
        plt.show()

def alexnet_2d_one_plot(result = "default", classes = 5):
    classes = int(classes)
    if result == "Acc-AlexNet":
        with open('Results\AlexNet2d\AccAlexNet2.txt', 'rb') as handle:
            a = pickle.load(handle)
        x = np.array(range(0, classes))
        y = np.zeros(classes)
        for i in range (len(a)):
          y[i]=a[i]   
        plt.title("Acc values")
        plt.xlabel("Classes")
        plt.ylabel("Acc values")
        plt.plot(x, y, color = "Orange", marker = "o", label = "Acc values")
        plt.grid()
        plt.legend()
        plt.show()
        
    elif result == "Auc-AlexNet":
        with open('Results\AlexNet2d\AucAlexNet2.txt', 'rb') as handle1:
            b = pickle.load(handle1)
        x = np.array(range(0, classes))
        y1 = np.zeros(classes)
        for i in range (len(b)):
          y1[i]=b[i]  
        plt.title("AlexNet - Auc values")
        plt.xlabel("Classes")
        plt.ylabel("Auc values")
        plt.plot(x, y1, color = "Green", marker = "o", label = "Auc values")
        plt.grid()    
        plt.legend()
        plt.show()

    
    elif result == "Npv-AlexNet":
        with open(rb'Results\AlexNet2d\NpvAlexNet2.txt', 'rb') as handle1:
            c = pickle.load(handle1)
        x = np.array(range(0, classes))
        y2 = np.zeros(classes)
        for i in range (len(c)):
          y2[i]=c[i]        
        plt.title("NPV values")
        plt.xlabel("Classes")
        plt.ylabel("NPV values")
        plt.plot(x, y2, color = "blue", marker = "o", label = "NPV values")
        plt.grid()        
        plt.legend()
        plt.show()
        
    elif result == "Ppv-AlexNet":
        with open('Results\AlexNet2d\PpvAlexNet2.txt', 'rb') as handle1:
            d = pickle.load(handle1)
        x = np.array(range(0, classes))
        y3 = np.zeros(classes)
        for i in range (len(d)):
          y3[i]=d[i]    
        plt.title("AlexNet - PPV valuess")
        plt.xlabel("Classes")
        plt.ylabel("PPV values")
        plt.plot(x, y3, color = "red", marker = "o", label = "PPV values")
        plt.grid()
        plt.legend()
        plt.show()
      
    elif result == "Sen-AlexNet":
        with open('Results\AlexNet2d\SenAlexNet2.txt', 'rb') as handle1:
            e = pickle.load(handle1)
        x = np.array(range(0, classes))
        y4 = np.zeros(10)
        for i in range (len(e)):
          y4[i]=e[i]     
        plt.title("AlexNet - Sen")
        plt.xlabel("Classes")
        plt.ylabel("Sen values")
        plt.plot(x, y4, color = "black", marker = "o", label = "Sen values")
        plt.grid()
        plt.legend()
        plt.show()

    elif result == "Spe-AlexNet":
        with open('Results\AlexNet2d\SenAlexNet2.txt', 'rb') as handle1:
            f = pickle.load(handle1)
        x = np.array(range(0, classes))
        y5 = np.zeros(classes)
        for i in range (len(f)):
          y5[i]=f[i]  
        plt.title("AlexNet - Spe")
        plt.xlabel("Classes")
        plt.ylabel("Spe values")
        plt.plot(x, y5, color = "Purple", marker = "o", label = "Spe values")
        plt.grid()
        plt.legend()
        plt.show()

def alexnet_2d_print(result = "default", classes=5):
    classes = int(classes)
    if result == "Acc-AlexNet":
        object_Acc_Generic = []
        with (open("Results\AlexNet2d\AccAlexNet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_Acc_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_Acc_Generic)   
                 
    elif result == "Auc-AlexNet":
        object_Auc_Generic = []
        with (open("Results\AlexNet2d\AucAlexNet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_Auc_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_Auc_Generic)
                 
    elif result == "Npv-AlexNet":
        object_NPV_Generic = []
        with (open(rb"Results\AlexNet2d\NpvAlexNet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_NPV_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_NPV_Generic)
                 
    elif result == "Ppv-AlexNet":
        object_PPV_Generic = []
        with (open("Results\AlexNet2d\PpvAlexNet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_PPV_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_PPV_Generic)
                 
    elif result == "Sen-AlexNet":
        object_SEN_Generic = []
        with (open("Results\AlexNet2d\SenAlexNet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_SEN_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_SEN_Generic)

    elif result == "Spe-AlexNet":
        object_SPE_Generic = []
        with (open("Results\AlexNet2d\SpeAlexNet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_SPE_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_SPE_Generic)

    else:    
            print("Select the result in the select box") 
    


# ----------------- Print 2d LeNet ------------------------

def lenet_2d_all_plot(result = "default", classes = 5):
        classes = int(classes)
        with open('Results\LeNet2d\AccLenet2.txt', 'rb') as handle:
            a = pickle.load(handle)
        x = np.array(range(0, classes))
        y = np.zeros(classes)
        for i in range (len(a)):
          y[i]=a[i]          
        with open('Results\LeNet2d\AucLeNet2.txt', 'rb') as handle1:
            b = pickle.load(handle1)
        x = np.array(range(0, classes))
        y1 = np.zeros(classes)
        for i in range (len(b)):
          y1[i]=b[i]          
        with open(rb'Results\LeNet2d\NpvLeNet2.txt', 'rb') as handle1:
            c = pickle.load(handle1)
        x = np.array(range(0, classes))
        y2 = np.zeros(classes)
        for i in range (len(b)):
          y2[i]=c[i]           
        with open('Results\LeNet2d\PpvLeNet2.txt', 'rb') as handle1:
            d = pickle.load(handle1)
        x = np.array(range(0, classes))
        y3 = np.zeros(classes)
        for i in range (len(b)):
          y3[i]=d[i]             
        with open('Results\LeNet2d\SenLeNet2.txt', 'rb') as handle1:
            e = pickle.load(handle1)
        x = np.array(range(0, classes))
        y4 = np.zeros(classes)
        for i in range (len(b)):
          y4[i]=e[i]           
        with open('Results\LeNet2d\SpeLeNet2.txt', 'rb') as handle1:
            f = pickle.load(handle1)
        x = np.array(range(0, classes))
        y5 = np.zeros(classes)
        for i in range (len(b)):
          y5[i]=f[i]    
        plt.title("LeNet - ACC | ACU | NPV | PPV | SPE | SEN")
        plt.xlabel("Classes")
        plt.ylabel("ACC | ACU | NPV | PPV | SPE | SEN")
        plt.plot(x, y, color = "Orange", marker = "o", label = "Acc values")
        plt.plot(x, y1, color = "Green", marker = "o", label = "Auc values")
        plt.plot(x, y2, color = "blue", marker = "o", label = "NPV values")
        plt.plot(x, y3, color = "red", marker = "o", label = "PPV values")
        plt.plot(x, y4, color = "black", marker = "o", label = "Sen values")
        plt.plot(x, y5, color = "Purple", marker = "o", label = "Spe values")
        plt.grid()
        plt.legend()
        plt.show()

def lenet_2d_one_plot(result = "default", classes = 5):
    classes = int(classes)
    if result == "Acc-LeNet":
        with open('Results\LeNet2d\AccLenet2.txt', 'rb') as handle:
            a = pickle.load(handle)
        x = np.array(range(0, classes))
        y = np.zeros(classes)
        for i in range (len(a)):
          y[i]=a[i]    
        plt.title("LeNet - Acc values")
        plt.xlabel("Classes")
        plt.ylabel("Acc values")
        plt.plot(x, y, color = "Orange", marker = "o", label = "Acc values")
        plt.grid()
        plt.legend()
        plt.show()
              
    elif result == "Auc-LeNet":
        with open('Results\LeNet2d\AucLenet2.txt', 'rb') as handle1:
            b = pickle.load(handle1)
        x = np.array(range(0, classes))
        y1 = np.zeros(classes)
        for i in range (len(b)):
          y1[i]=b[i] 
        plt.title("LeNet - Auc values")
        plt.xlabel("Classes")
        plt.ylabel("Auc values")
        plt.plot(x, y1, color = "Green", marker = "o", label = "Auc values")
        plt.grid()
        plt.legend()
        plt.show()
        
    elif result == "Npv-LeNet":
        with open(rb'Results\LeNet2d\NpvLenet2.txt', 'rb') as handle1:
            c = pickle.load(handle1)
        x = np.array(range(0, classes))
        y2 = np.zeros(classes)
        for i in range (len(c)):
          y2[i]=c[i]      
        plt.title("LeNet - NPV values")
        plt.xlabel("Classes")
        plt.ylabel("NPV values")
        plt.plot(x, y2, color = "blue", marker = "o", label = "NPV values")
        plt.grid()
        plt.legend()
        plt.show()    
        
    elif result == "Ppv-LeNet":
        with open('Results\LeNet2d\PpvLenet2.txt', 'rb') as handle1:
            d = pickle.load(handle1)
        x = np.array(range(0, classes))
        y3 = np.zeros(classes)
        for i in range (len(d)):
          y3[i]=d[i]   
        plt.title("LeNet - PPV values")
        plt.xlabel("Classes")
        plt.ylabel("PPV values")
        plt.plot(x, y3, color = "red", marker = "o", label = "PPV values")
        plt.grid()
        plt.legend()
        plt.show()    
        
    elif result == "Sen-LeNet":
        with open('Results\LeNet2d\SenLenet2.txt', 'rb') as handle1:
            e = pickle.load(handle1)
        x = np.array(range(0, classes))
        y4 = np.zeros(classes)
        for i in range (len(e)):
          y4[i]=e[i]      
        plt.title("LeNet - Sen values")
        plt.xlabel("Classes")
        plt.ylabel("Sen values")
        plt.plot(x, y4, color = "black", marker = "o", label = "Sen values")
        plt.grid()
        plt.legend()
        plt.show()

    elif result == "Spe-LeNet":
        with open('Results\LeNet2d\SpeLenet2.txt', 'rb') as handle1:
            f = pickle.load(handle1)
        x = np.array(range(0, classes))
        y5 = np.zeros(classes)
        for i in range (len(f)):
          y5[i]=f[i]    
        plt.xlabel("Classes")
        plt.ylabel("Spe values")
        plt.plot(x, y5, color = "Purple", marker = "o", label = "Spe values")
        plt.grid()
        plt.legend()
        plt.show()

def lenet_2d_print(result = "default" , classes = 5):
    if result == "Acc-LeNet":
        object_Acc_Generic = []
        with (open("Results\LeNet2d\AccLenet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_Acc_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_Acc_Generic)   
                 
    elif result == "Auc-LeNet":
        object_Auc_Generic = []
        with (open("Results\LeNet2d\AucLenet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_Auc_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_Auc_Generic)
                 
    elif result == "Npv-LeNet":
        object_NPV_Generic = []
        with (open(rb"Results\LeNet2d\NpvLenet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_NPV_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_NPV_Generic)
                 
    elif result == "Ppv-LeNet":
        object_PPV_Generic = []
        with (open("Results\LeNet2d\PpvLenet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_PPV_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_PPV_Generic)
                 
    elif result == "Sen-LeNet":
        object_SEN_Generic = []
        with (open("Results\LeNet2d\SenLenet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_SEN_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_SEN_Generic)

    elif result == "Spe-LeNet":
        object_SPE_Generic = []
        with (open("Results\LeNet2d\SpeLenet2.txt","rb")) as openfile:
              while True:
                 try:
                    object_SPE_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_SPE_Generic)

    else:    
            print("Select the result in the select box") 
            
                     

# ----------------------- Print 2d Edit---------------

def edit_2d_all_plot(result = "default", classes = 5):
        classes = int(classes)
        with open('Results\Edit2d\AccEdit2d.txt', 'rb') as handle:
            a = pickle.load(handle)
        x = np.array(range(0, classes))
        y = np.zeros(classes)
        for i in range (len(a)):
          y[i]=a[i] 
        with open('Results\Edit2d\AucEdit2d.txt', 'rb') as handle1:
            b = pickle.load(handle1)
        x = np.array(range(0, classes))
        y1 = np.zeros(classes)
        for i in range (len(b)):
          y1[i]=b[i]         
        with open(rb'Results\Edit2d\PpvEdit2d.txt', 'rb') as handle2:
            c = pickle.load(handle2)
        x = np.array(range(0, classes))
        y2 = np.zeros(classes)
        for i in range (len(c)):
          y2[i]=c[i]      
        with open(r'Results\Edit2d\NpvEdit2d.txt', 'rb') as handle3:
            d = pickle.load(handle3)
        x = np.array(range(0, classes))
        y3 = np.zeros(classes)
        for i in range (len(d)):
          y3[i]=d[i]     
        with open('Results\Edit2d\SenEdit2d.txt', 'rb') as handle4:
            e = pickle.load(handle4)
        x = np.array(range(0, classes))
        y4 = np.zeros(classes)
        for i in range (len(e)):
          y4[i]=e[i] 
        with open('Results\Edit2d\SpeEdit2d.txt', 'rb') as handle5:
            f = pickle.load(handle5)
        x = np.array(range(0, classes))
        y5 = np.zeros(classes)
        for i in range (len(f)):
          y5[i]=f[i]    
        plt.title("Edit Results - ACC | ACU | NPV | PPV | SPE | SEN")
        plt.xlabel("Classes")
        plt.ylabel("ACC | ACU | NPV | PPV | SPE | SEN")
        plt.plot(x, y, color = "Orange", marker = "o", label = "Acc values")
        plt.plot(x, y1, color = "Green", marker = "o", label = "Auc values")
        plt.plot(x, y2, color = "blue", marker = "o", label = "NPV values")
        plt.plot(x, y3, color = "red", marker = "o", label = "PPV values")
        plt.plot(x, y4, color = "black", marker = "o", label = "Sen values")
        plt.plot(x, y5, color = "Purple", marker = "o", label = "Spe values")
        plt.grid()
        plt.legend()
        plt.show()

def edit_2d_one_plot(result = "default" , classes = 5):
    classes = int(classes)
    if result == "Acc-Edit2d":
        with open('Results\Edit2d\AccEdit2d.txt', 'rb') as handle:
            a = pickle.load(handle)
        x = np.array(range(0, classes))
        y = np.zeros(classes)
        for i in range (len(a)):
          y[i]=a[i] 
        plt.title("LeNet - Acc values")
        plt.xlabel("Classes")
        plt.ylabel("Acc values")
        plt.plot(x, y, color = "Orange", marker = "o", label = "Acc values")
        plt.grid()
        plt.legend()
        plt.show()
    
    elif result == "Auc-Edit2d":
        with open('Results\Edit2d\AucEdit2d.txt', 'rb') as handle1:
            b = pickle.load(handle1)
        x = np.array(range(0, classes))
        y1 = np.zeros(classes)
        for i in range (len(b)):
          y1[i]=b[i]      
        plt.title("LeNet - Auc values")
        plt.xlabel("Classes")
        plt.ylabel("Auc values")
        plt.plot(x, y1, color = "Green", marker = "o", label = "Auc values")
        plt.grid()
        plt.legend()
        plt.show()
        
    elif result == "Npv-Edit2d":
        with open(rb'Results\Edit2d\NpvEdit2d.txt', 'rb') as handle2:
            c = pickle.load(handle2)
        x = np.array(range(0, classes))
        y2 = np.zeros(classes)
        for i in range (len(c)):
          y2[i]=c[i]  
        plt.title("LeNet - NPV values")
        plt.xlabel("Classes")
        plt.ylabel("NPV values")
        plt.plot(x, y2, color = "blue", marker = "o", label = "NPV values")
        plt.grid()
        plt.legend()
        plt.show()    
        
    elif result == "Ppv-Edit2d":
        with open('Results\Edit2d\PpvEdit2d.txt', 'rb') as handle3:
            d = pickle.load(handle3)
        x = np.array(range(0, classes))
        y3 = np.zeros(classes)
        for i in range (len(d)):
          y3[i]=d[i]    
        plt.title("LeNet - PPV values")
        plt.xlabel("Classes")
        plt.ylabel("PPV values")
        plt.plot(x, y3, color = "red", marker = "o", label = "PPV values")
        plt.grid()
        plt.legend()
        plt.show()    
        
    elif result == "Sen-Edit2d":
        with open('Results\Edit2d\SenEdit2d.txt', 'rb') as handle4:
            e = pickle.load(handle4)
        x = np.array(range(0, classes))
        y4 = np.zeros(classes)
        for i in range (len(e)):
          y4[i]=e[i] 
        plt.title("LeNet - Sen values")
        plt.xlabel("Classes")
        plt.ylabel("Sen values")
        plt.plot(x, y4, color = "black", marker = "o", label = "Sen values")
        plt.grid()
        plt.legend()
        plt.show()

    elif result == "Spe-Edit2d":
        with open('Results\Edit2d\SpeEdit2d.txt', 'rb') as handle5:
            f = pickle.load(handle5)
        x = np.array(range(0, classes))
        y5 = np.zeros(classes)
        for i in range (len(f)):
          y5[i]=f[i]    
        plt.title("LeNet - Spe values")
        plt.xlabel("Classes")
        plt.ylabel("Spe values")
        plt.plot(x, y5, color = "Purple", marker = "o", label = "Spe values")
        plt.grid()
        plt.legend()
        plt.show()
       
def edit_2d_print(result = "default" , classes = 5):
    if result == "Acc-Edit2d":
        object_Acc_Generic = []
        with (open("Results\Edit2d\AccEdit2d.txt","rb")) as openfile:
              while True:
                 try:
                    object_Acc_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_Acc_Generic)   
                 
    elif result == "Auc-Edit2d":
        object_Auc_Generic = []
        with (open("Results\Edit2d\AucEdit2d.txt","rb")) as openfile:
              while True:
                 try:
                    object_Auc_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_Auc_Generic)
                 
    elif result == "Npv-Edit2d":
        object_NPV_Generic = []
        with (open(rb"Results\Edit2d\NpvEdit2d.txt","rb")) as openfile:
              while True:
                 try:
                    object_NPV_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_NPV_Generic)
                 
    elif result == "Ppv-Edit2d":
        object_PPV_Generic = []
        with (open("Results\Edit2d\PpvEdit2d.txt","rb")) as openfile:
              while True:
                 try:
                    object_PPV_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_PPV_Generic)
                 
    elif result == "Sen-Edit2d":
        object_SEN_Generic = []
        with (open("Results\Edit2d\SenEdit2d.txt","rb")) as openfile:
              while True:
                 try:
                    object_SEN_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_SEN_Generic)

    elif result == "Spe-Edit2d":
        object_SPE_Generic = []
        with (open("Results\Edit2d\SpeEdit2d.txt","rb")) as openfile:
              while True:
                 try:
                    object_SPE_Generic.append(pickle.load(openfile))
                 except EOFError:
                     break
                 print(object_SPE_Generic)

    else:    
            print("Select the result in the select box") 
            

