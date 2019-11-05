# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:33:26 2019

@author: Administrator
"""

from PIL import Image
import glob
#glob.glob --> list pour parcourir toutes les images
import numpy as np

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn import svm


path= 'C:/TP-ML/yalefaces/subject01.centerlight'

#Partie1
img=Image.open(path)
img1=np.array(img)
img2=img1.reshape(img1.shape[0]*img1.shape[1])
data=[]
paths=glob.glob('C:/TP-ML/yalefaces/subject*')
for i in paths:
    img=Image.open(i)
    img1=np.array(img)
    img2=img1.reshape(img1.shape[0]*img1.shape[1])
    data.append(img2)
data1=np.array(data)

pca=PCA(n_components=165)
X1=pca.fit_transform(data1)
Y=np.repeat(range(1,16),11)
x_train,x_test,y_train,y_test=train_test_split(X1,Y,test_size=0.33, random_state=0)
    
model=svm.SVC(kernel='linear',C=1)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
acc165=accuracy_score(y_test,y_pred)

pca=PCA(n_components=100)
X1=pca.fit_transform(data1)
Y=np.repeat(range(1,16),11)
x_train,x_test,y_train,y_test=train_test_split(X1,Y,test_size=0.33, random_state=0)
    
model=svm.SVC(kernel='linear',C=1)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
acc100=accuracy_score(y_test,y_pred)

pca=PCA(n_components=50)
X1=pca.fit_transform(data1)
Y=np.repeat(range(1,16),11)
x_train,x_test,y_train,y_test=train_test_split(X1,Y,test_size=0.33, random_state=0)
    
model=svm.SVC(kernel='linear',C=1)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
acc50=accuracy_score(y_test,y_pred)

pca=PCA(n_components=25)
X1=pca.fit_transform(data1)
Y=np.repeat(range(1,16),11)
x_train,x_test,y_train,y_test=train_test_split(X1,Y,test_size=0.33, random_state=0)
    
model=svm.SVC(kernel='linear',C=1)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
acc25=accuracy_score(y_test,y_pred)

pca=PCA(n_components=15)
X1=pca.fit_transform(data1)
Y=np.repeat(range(1,16),11)
x_train,x_test,y_train,y_test=train_test_split(X1,Y,test_size=0.33, random_state=0)
    
model=svm.SVC(kernel='linear',C=1)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
acc15=accuracy_score(y_test,y_pred)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
model=LDA()
Y=np.repeat(range(1,16),11)
X2= model.fit_transform(data1,Y)
x_train,x_test,y_train,y_test=train_test_split(X2,Y,test_size=0.33)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
acc_LDa=accuracy_score(y_test,y_pred)


