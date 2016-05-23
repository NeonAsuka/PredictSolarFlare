###
#
#Author: Ran Cheng
###


import numpy as np

#####################################################################################
#This part is used to load dataset and pre-process it

#load raw data, raw data file's name is "data1.txt"
sc = np.genfromtxt('data1.txt',dtype='str')

#Because the features of first three column is represented by char, we have to transform them into int number
#transform the first column's feature values into int number: A=0,B=1,C=2...
#first colume
c1 = sc[:,0]
for i in range(c1.size):
    if c1[i]=='A':
        sc[i,0]=0
    elif c1[i]=='B':
        sc[i,0]=1
    elif c1[i]=='C':
        sc[i,0]=2
    elif c1[i]=='D':
        sc[i,0]=3
    elif c1[i]=='E':
        sc[i,0]=4
    elif c1[i]=='F':
        sc[i,0]=5
    elif c1[i]=='H':
        sc[i,0]=6

#transform the second column's feature values into int number: X=0,R=1,S=2...
c2 = sc[:,1]
for i in range(c2.size):
    if c2[i]=='X':
        sc[i,1]=0
    elif c2[i]=='R':
        sc[i,1]=1
    elif c2[i]=='S':
        sc[i,1]=2
    elif c2[i]=='A':
        sc[i,1]=3
    elif c2[i]=='H':
        sc[i,1]=4
    elif c2[i]=='K':
        sc[i,1]=5

#transform the third column's feature values into int number: X=0,O=1,I=2,C=3
c3 = sc[:,2]
for i in range(c3.size):
    if c3[i]=='X':
        sc[i,2]=0
    elif c3[i]=='O':
        sc[i,2]=1
    elif c3[i]=='I':
        sc[i,2]=2
    elif c3[i]=='C':
        sc[i,2]=3

#but our array still is string type
#transform string array into int np array, sc is the object array 
def myInt(myList):
    return map(int,myList)
sc = map(myInt,sc)
sc = np.array(sc)

#splite the array into feature section and label sections
#label_scm used for multi-task learning
#label_sc1,2,3 used for seperate learning 
features_sc = sc[:,0:10]
label_scm = sc[:,10:13]
label_sc1 = sc[:,10]
label_sc2 = sc[:,11]
label_sc3 = sc[:,12]


######################################################################
#this part is random forest regressor's pipeline
#this code is just pipeline or label_sc1. piplelines of label_sc2,label_sc3 are very similar, so I don't list them. 

#load necessray libs
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split

#set random seed to make results stable
np.random.seed(1)

#splite dataset to get necessary sub-dataset 
features_train, features_test, labels_train, labels_test = train_test_split(features_sc,label_sc1,test_size=0.33,random_state=42)

#pre-process: dimensional reduction(SVD) and feature selection(SlestKBest)
svd1 = TruncatedSVD(n_components=9,random_state=1).fit(features_train)
svd_temp1 = svd1.transform(features_train)
model1 = SelectKBest(k='all')
model1.fit(svd_temp1,labels_train)
features_train = model1.transform(svd_temp1)

svd2 = TruncatedSVD(n_components=9,random_state=1).fit(features_test)
svd_temp2 = svd2.transform(features_test)
model2 = SelectKBest(k='all')
model2.fit(svd_temp2,labels_test)
features_test = model2.transform(svd_temp2)

#set values of hyper-parameter n_components
hyperparameters=np.array([10,15,20,25,30,35,40])

bestscore=0
besthyper=0
fold=3

#iterator to get the optimal hyper-parameter 
for parameters in hyperparameters:
	#prepare random forest and k-fold cross-validation models
    mid = RandomForestRegressor(n_estimators=parameters,random_state=1)
    kf = KFold(len(features_test), n_folds=fold,shuffle=True)
         
    totalscore=0
        
    for train_index, test_index in kf:
        X_train, X_test = features_train[train_index], features_test[test_index]
        y_train, y_test = features_train[train_index], features_test[test_index] 
    
        #train regression model and get score 
        clf = mid.fit(X_train, y_train)
        score= clf.score(X_test, y_test)
        totalscore=totalscore+score
    #calculate average score    
    ave=totalscore/fold 
    
    if (ave>bestscore):
        bestscore=ave
        besthyper=parameters

#print "best ave score:",bestscore
print "best hyperparameter:",besthyper

##############################################################################
#this part is Multi-task Lasso pipeline

#load necessary libs 
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import MultiTaskLasso
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split

#set random seed to make results stable
np.random.seed(1)

#splite dataset to get necessary sub-dataset
features_train, features_test, labels_train, labels_test = train_test_split(features_sc,label_scm,test_size=0.33,random_state=42)

#pre-process: dimensional reduction(SVD) 
svd1 = TruncatedSVD(n_components=9,random_state=1).fit(features_train)
features_train = svd1.transform(features_train)

svd2 = TruncatedSVD(n_components=9,random_state=1).fit(features_test)
features_test = svd2.transform(features_test)

#set values of hyper-parameter alpha
hyperparameters=np.array([0.000000001,0.01,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.4,0.6,0.8])

bestscore=0
besthyper=0
fold=3

#iterator to get the optimal hyper-parameter 
for parameters in hyperparameters:
	#prepare random forest and k-fold cross-validation models
    mid = MultiTaskLasso(alpha=parameters,random_state=1)
    kf = KFold(len(features_test), n_folds=fold,shuffle=True)
         
    totalscore=0
        
    for train_index, test_index in kf:
        X_train, X_test = features_train[train_index], features_test[test_index]
        y_train, y_test = features_train[train_index], features_test[test_index] 
    
        #train regression model and get score
        clf = mid.fit(X_train, y_train)
        score= clf.score(X_test, y_test)
        totalscore=totalscore+score
        
    ave=totalscore/fold 
    if (ave>bestscore):
        bestscore=ave
        besthyper=parameters

#print "best ave score:",bestscore
print "best hyperparameter:",besthyper

#####################################################################
#this part is Multi-task Elastic-net pipeline

#load necessary libs 
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import MultiTaskElasticNet
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split

#set random seed to make results stable
np.random.seed(1)

#splite dataset to get necessary sub-dataset
features_train, features_test, labels_train, labels_test = train_test_split(features_sc,label_scm,test_size=0.33,random_state=42)

#pre-process: dimensional reduction(SVD)
svd1 = TruncatedSVD(n_components=9,random_state=1).fit(features_train)
features_train = svd1.transform(features_train)

svd2 = TruncatedSVD(n_components=9,random_state=1).fit(features_test)
features_test = svd2.transform(features_test)

#set values of hyper-parameter alpha, l1_ratio 
hyperparameters=np.array([0.000000001,0.01,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.4,0.6,0.8])
l1ratio=np.array([0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.4,0.6,0.8])

bestscore=0
besthyper=0
bestratio=0
fold=3

#iterator to get the optimal hyper-parameter 
for ratio in l1ratio:
    for parameters in hyperparameters:
		#prepare random forest and k-fold cross-validation models
        mid = MultiTaskElasticNet(alpha=parameters,l1_ratio=ratio,random_state=1)
        kf = KFold(len(features_test), n_folds=fold,shuffle=True)
         
        totalscore=0
        
        for train_index, test_index in kf:
            X_train, X_test = features_train[train_index], features_test[test_index]
            y_train, y_test = features_train[train_index], features_test[test_index] 
    
			#train regression model and get score
            clf = mid.fit(X_train, y_train)
            score= clf.score(X_test, y_test)
            totalscore=totalscore+score
        
        ave=totalscore/fold 
        if (ave>bestscore):
            bestscore=ave
            besthyper=parameters
            bestratio=ratio

#print "best ave score:",bestscore
print "best hyperparameter:",besthyper
print "best ratio:",bestratio

############################################################
#this part is used to calculate the Random Forest Regressor's score when the hyper-parameter is optimal 

#load necessary libs 
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split

#splite dataset to get necessary sub-dataset
features_train, features_test, labels_train, labels_test = train_test_split(features_sc,label_sc3,test_size=0.33,random_state=42)

#pre-process: dimensional reduction(SVD) and feature selection(SlestKBest)
svd1 = TruncatedSVD(n_components=9,random_state=1).fit(features_train)
svd_temp1 = svd1.transform(features_train)
model1 = SelectKBest(k='all')
model1.fit(svd_temp1,labels_train)
features_train = model1.transform(svd_temp1)

svd2 = TruncatedSVD(n_components=9,random_state=1).fit(features_test)
svd_temp2 = svd2.transform(features_test)
model2 = SelectKBest(k='all')
model2.fit(svd_temp2,labels_test)
features_test = model2.transform(svd_temp2)

#do regression 
rf = RandomForestRegressor(n_estimators=40,random_state=1)
rf.fit(features_train,labels_train)
print "rf",rf.score(features_test,labels_test)

##################################################################
#this part is used to calculate the Multi-Task Lasso's score when the hyper-parameter is optimal 

#load necessary libs 
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import MultiTaskLasso
from sklearn.cross_validation import train_test_split

#splite dataset to get necessary sub-dataset
features_train, features_test, labels_train, labels_test = train_test_split(features_sc,label_scm,test_size=0.33,random_state=42)

#pre-process: dimensional reduction(SVD)
svd1 = TruncatedSVD(n_components=9,random_state=1).fit(features_train)
features_train = svd1.transform(features_train)

svd2 = TruncatedSVD(n_components=9,random_state=1).fit(features_test)
features_test = svd2.transform(features_test)

#do regression
mtl = MultiTaskLasso(alpha=0.000000001,random_state=1)
mtl.fit(features_train,labels_train)
print "MultiTaskLasso",mtl.score(features_test,labels_test)

######################################################################
#this part is used to calculate the Multi-Task Elastic-net's score when the hyper-parameter is optimal 

#load necessary libs 
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.cross_validation import train_test_split

#splite dataset to get necessary sub-dataset
features_train, features_test, labels_train, labels_test = train_test_split(features_sc,label_scm,test_size=0.33,random_state=42)

#pre-process: dimensional reduction(SVD)
svd1 = TruncatedSVD(n_components=9,random_state=1).fit(features_train)
features_train = svd1.transform(features_train)

svd2 = TruncatedSVD(n_components=9,random_state=1).fit(features_test)
features_test = svd2.transform(features_test)

#do regression
mte = MultiTaskElasticNet(alpha=0.000000001,l1_ratio=0.01,random_state=1)
mte.fit(features_train,labels_train)
print "MultiTaskElasticNet",mte.score(features_test,labels_test)
##########################################################################

#All of the codes end. 
#Thank you!




