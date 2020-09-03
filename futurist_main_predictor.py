"""
    A simple neural network written in Keras (TensorFlow backend) to classify the IRIS data
"""

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

#from keras.models import Sequential
#from keras.layers import Dense,Conv1D,Dropout
#from keras.optimizers import Adam
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch,Hyperband
from tensorflow.keras import regularizers
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold
#from keras.callbacks import TensorBoard
from sklearn.metrics import cohen_kappa_score,confusion_matrix

import sklearn.ensemble


import lime
import lime.lime_tabular

import matplotlib.pyplot as plt
#plt.style.use('ggplot')
import scipy.stats


#iris_data = load_iris() # load the iris dataset
#
#print('Example data: ')
#print(iris_data.data[:5])
#print('Example labels: ')
#print(iris_data.target[:5])
#
#x = iris_data.data
#y_ = iris_data.target.reshape(-1, 1) # Convert data to a single column
#
## One Hot encode the class labels
#encoder = OneHotEncoder(sparse=False)
#y = encoder.fit_transform(y_)
##print(y)

dirpath = "/home/saswms/ShahryarWork/TemperatureGrand/Automated_classifier/"

def calc_Ncoeff(dfx):
	dfx['Ncoeff'] = dfx.res_cap/(dfx.height*dfx.res_area)   # 2*math.log(2)*
	return dfx

def calc_tempinc(dfx, inc=0):
	dfx['airtempwrm_inc'] = dfx.airtempwrm + inc #
	dfx['airtempcld_inc'] = dfx.airtempcld + inc
	return dfx
	
	
	
### INPUT DATA
iris = pd.read_csv(dirpath + "usgs_up_down_ai.csv")
iris = calc_Ncoeff(iris)

pred="meanerrWRMonly"#"strCLD"
predmek="meanerrWRMonly"#"strCLD"
#pred="meanerrCLDonly"#"strCLD"
#predmek="meanerrCLDonly"#"strCLD"

INC = 6

#print(iris['strWRM'])
iris.iloc[:,1:12] = iris.iloc[:,1:12].astype(np.float32)

#iris["strWRM"] = iris["strWRM"].map({"'Q1'":0,"'Q2'":1,"'Q3'":2,"'Q4'":3})

from sklearn.preprocessing import LabelBinarizer
species_lb = LabelBinarizer()
#Y_all = species_lb.fit_transform(iris[pred].values)
#Y_all = iris[pred].values
from sklearn.preprocessing import normalize
FEATURES = ['height','res_area', 'res_cap','koppen','Ncoeff', 'airtempwrm','ELEV_MASL']  #airtempwrm_inc

#
#X_all = iris[FEATURES].as_matrix()
#X_all = normalize(X_all)
#Y_all = (Y_all.reshape(-1, 1))
#
#
#Y = Y_all[:85,]
#X_data = X_all[:85,:]
#
#Y_test = Y_all[85:,]
#X_test= X_all[85:,:]
#


### PREDICITONS On Mekong
iris_mekplanned = pd.read_csv("/home/saswms/ShahryarWork/TemperatureGrand/Mekong_planned/mekong_planned_dams.csv")
iris_mekplanned = calc_Ncoeff(iris_mekplanned)
iris_mekplanned = calc_tempinc(iris_mekplanned, INC)

#iris_mek = iris_mek.dropna()
iris_mekplanned.iloc[:,4:30] = iris_mekplanned.iloc[:,4:30].astype(np.float32)
ID_mekplanned = iris_mekplanned['ID'].values
NAME_mekplanned = iris_mekplanned['Name'].values



X_mekplanned = iris_mekplanned[FEATURES].as_matrix()
X_mekplanned = normalize(X_mekplanned)

### PREDICITONS On Mekong Tributaries
iris_mekplannedt = pd.read_csv("/home/saswms/ShahryarWork/TemperatureGrand/Mekong_planned_tributaries/mekong_tributaries_planned_fillednew.csv")
iris_mekplannedt = calc_Ncoeff(iris_mekplannedt)
iris_mekplannedt = calc_tempinc(iris_mekplannedt, INC)

#iris_mek = iris_mek.dropna()
iris_mekplannedt.iloc[:,4:30] = iris_mekplannedt.iloc[:,4:30].astype(np.float32)
ID_mekplannedt = iris_mekplannedt['idnum'].values
NAME_mekplannedt = iris_mekplannedt['Name'].values

X_mekplannedt = iris_mekplannedt[FEATURES].as_matrix()
X_mekplannedt = normalize(X_mekplannedt)


### PREDICTONS On World planned
iris_worldplanned = pd.read_csv("/home/saswms/ShahryarWork/TemperatureGrand/World_planned/inputs_world.csv")
iris_worldplanned = iris_worldplanned.iloc[:,:12]
iris_worldplanned = calc_Ncoeff(iris_worldplanned)
iris_worldplanned = calc_tempinc(iris_worldplanned, INC)

iris_worldplanned = iris_worldplanned.dropna()
iris_worldplanned.iloc[:,3:11] = iris_worldplanned.iloc[:,3:11].astype(np.float32)
ID_worldplanned = iris_worldplanned['idnum'].values
NAME_worldplanned = iris_worldplanned['Name'].values

X_worldplanned = iris_worldplanned[FEATURES].as_matrix()
X_worldplanned = normalize(X_worldplanned)


#loaded_model = tf.keras.models.load_model('models/reg_mek80perc_Ncoeff')
loaded_model = tf.keras.models.load_model('models/reg_mek80perc_NcoeffCLD')
##
model=loaded_model
#%% Mek PLANNED



Y_mekpred = model.predict(X_mekplanned) #

#truth = Y_testind#mek #val_y # tf.argmax(Y_test, axis=1, output_type=tf.int32)
prediction =  Y_mekpred #tf.argmax(Y_pred, axis=1, output_type=tf.int32)
#print('t',truth) #.eval(session=tf.compat.v1.Session()))
print('p',prediction) #.eval(session=tf.compat.v1.Session()))

binc = pd.IntervalIndex.from_tuples([(-20, -6), (-5.9999,-0),(0.0001,5),(5.001,15)])
#binc = pd.IntervalIndex.from_tuples([(-20,-0),(0.0001,15)])

labelc=[1,2,3,4]
d = dict(zip(binc,labelc))
ytest_ctg=[]
ypred_ctg=[]
tpm = np.zeros([len(prediction),3])
#j=0
#for i in pd.cut(truth.flatten(), bins=binc, labels=labelc).tolist():
#    ytest_ctg.append(d.get(i))
#    tp[j,0] = ID_testind[j] #mek[j] #				
#    tp[j,1] = truth.flatten()[j]
#    j=j+1

k=0   
for i in pd.cut(prediction.flatten(), bins=binc, labels=labelc).tolist():
    ypred_ctg.append(d.get(i))
    tpm[k,0] = ID_mekplanned[k]
    tpm[k,1] = prediction.flatten()[k]
    tpm[k,2] = d.get(i)
    k=k+1
    

#print('t',ytest_ctg)
#print('p',ypred_ctg)


#fig = plt.figure(figsize=(6.5,4)) # Create matplotlib figure
#
#ax = fig.add_subplot(111) # Create matplotlib axes
##ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
#
#width = 0.4
#my_colors = ['b','b','b','deepskyblue','orange','orange','orange','deepskyblue','r']
##list('rgbkymc')  #red, green, blue, black, etc.
#
##iris_mekplanned['res_cap'].plot(kind='bar', color= my_colors, ax=ax, width=width, position=1,label='capacity')
#iris_mekplanned['res_area'].plot(kind='bar', color=my_colors, ax=ax, width=width, position=0,label='area')
##
#ax.set_ylabel('Res area (km^2)')
#ax.set_xticklabels(iris_mekplanned['Name'].values)
#plt.show()








#%% Mek TRIBUTARIES PLANNED

Y_mekpredt = model.predict(X_mekplannedt) #

#truth = Y_testind#mek #val_y # tf.argmax(Y_test, axis=1, output_type=tf.int32)
prediction =  Y_mekpredt #tf.argmax(Y_pred, axis=1, output_type=tf.int32)
#print('t',truth) #.eval(session=tf.compat.v1.Session()))
#print('p',prediction) #.eval(session=tf.compat.v1.Session()))

binc = pd.IntervalIndex.from_tuples([(-20, -6), (-5.9999,-0),(0.0001,5),(5.001,15)])
#binc = pd.IntervalIndex.from_tuples([(-20,-0),(0.0001,15)])

labelc=[1,2,3,4]
d = dict(zip(binc,labelc))
ytest_ctg=[]
ypred_ctg=[]
tpt = np.zeros([len(prediction),3])
#j=0
#for i in pd.cut(truth.flatten(), bins=binc, labels=labelc).tolist():
#    ytest_ctg.append(d.get(i))
#    tp[j,0] = ID_testind[j] #mek[j] #				
#    tp[j,1] = truth.flatten()[j]
#    j=j+1

k=0   
for i in pd.cut(prediction.flatten(), bins=binc, labels=labelc).tolist():
    ypred_ctg.append(d.get(i))
    tpt[k,0] = ID_mekplannedt[k]
    tpt[k,1] = prediction.flatten()[k]
    tpt[k,2] = d.get(i)
    k=k+1
    
#print(tpt)


#tpmall = np.vstack([tpt,tpm])
#print(tpmall)

#fig = plt.figure(figsize=(6.5,4)) # Create matplotlib figure

#ax = fig.add_subplot(111) # Create matplotlib axes
##ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
#
#width = 0.4
#my_colors = ['b','b','b','deepskyblue','orange','orange','orange','deepskyblue','r']
##list('rgbkymc')  #red, green, blue, black, etc.

##iris_mekplanned['res_cap'].plot(kind='bar', color= my_colors, ax=ax, width=width, position=1,label='capacity')
#iris_mekplannedt['res_area'].plot(kind='bar', color=my_colors, ax=ax, width=width, position=0,label='area')
##
#ax.set_ylabel('Res area (km^2)')
#ax.set_xticklabels(iris_mekplannedt['Name'].values)
#plt.show()


#%% WORLD PLANNED

Y_worldpred = model.predict(X_worldplanned) #

#truth = Y_testind#mek #val_y # tf.argmax(Y_test, axis=1, output_type=tf.int32)
prediction =  Y_worldpred #tf.argmax(Y_pred, axis=1, output_type=tf.int32)
#print('t',truth) #.eval(session=tf.compat.v1.Session()))
#print('p',prediction) #.eval(session=tf.compat.v1.Session()))

binc = pd.IntervalIndex.from_tuples([(-20, -6), (-5.9999,-0),(0.0001,5),(5.001,15)])
#binc = pd.IntervalIndex.from_tuples([(-20,-0),(0.0001,15)])

labelc=[1,2,3,4]
d = dict(zip(binc,labelc))
ytest_ctg=[]
ypred_ctg=[]
tp = np.zeros([len(prediction),3])
#j=0
#for i in pd.cut(truth.flatten(), bins=binc, labels=labelc).tolist():
#    ytest_ctg.append(d.get(i))
#    tp[j,0] = ID_testind[j] #mek[j] #				
#    tp[j,1] = truth.flatten()[j]
#    j=j+1

k=0   
for i in pd.cut(prediction.flatten(), bins=binc, labels=labelc).tolist():
    ypred_ctg.append(d.get(i))
    tp[k,0] = ID_worldplanned[k]
    tp[k,1] = prediction.flatten()[k]
    tp[k,2] = d.get(i)
    k=k+1
    
#print(tp)
tpall = np.vstack([tp,tpt,tpm])
print(tpall)

#print('t',ytest_ctg)
#print('p',ypred_ctg)


#fig = plt.figure(figsize=(6.5,4)) # Create matplotlib figure
#
#ax = fig.add_subplot(111) # Create matplotlib axes
##ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
#
#width = 0.4
#my_colors = ['b','b','b','deepskyblue','orange','orange','orange','deepskyblue','r']
##list('rgbkymc')  #red, green, blue, black, etc.
#
##iris_mekplanned['res_cap'].plot(kind='bar', color= my_colors, ax=ax, width=width, position=1,label='capacity')
#iris_worldplanned['res_area'].plot(kind='bar', color=my_colors, ax=ax, width=width, position=0,label='area')
##
#ax.set_ylabel('Res area (km^2)')
#ax.set_xticklabels(iris_worldplanned['Name'].values)
#plt.show()
#%% lime		

## Mekong planned pred
#explainer = lime.lime_tabular.LimeTabularExplainer(X_mekplanned, feature_names=FEATURES,
#								class_names=[pred], 
#								verbose=True, mode='regression',
#								discretize_continuous=False)
#for i in range(9): #[2,5,8,10,17,1x8,19]:
#	print(ID_mekplanned[i],NAME_mekplanned[i])
#	exp = explainer.explain_instance(X_mekplanned[i], model.predict, num_features=7)
#	exp.as_pyplot_figure(titleid=NAME_mekplanned[i])
##	my_pyplot_figure(exp,ID_testmek[i])
#


### World planned pred
explainer = lime.lime_tabular.LimeTabularExplainer(X_worldplanned, feature_names=FEATURES,
								class_names=[pred], 
								verbose=True, mode='regression',
								discretize_continuous=False)
for i in range(200): #[2,5,8,10,17,1x8,19]:
	print(ID_worldplanned[i],NAME_worldplanned[i])
	exp = explainer.explain_instance(X_worldplanned[i], model.predict, num_features=7)
	exp.as_pyplot_figure(titleid=NAME_worldplanned[i])
#	my_pyplot_figure(exp,ID_testmek[i])


