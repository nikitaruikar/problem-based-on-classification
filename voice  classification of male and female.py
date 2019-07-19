#Importing dataset
import numpy as np
import pandas as pd
import seaborn as sns

dataset1 = pd.read_csv('D:\\cognitior\Basics of data science\\dataset\\voice.csv')

dataset1.isnull().sum()

x = dataset1.iloc[:,:-1].values
pd.DataFrame(x)

y = dataset1.iloc[:,-1].values
pd.DataFrame(y)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

#Divide the data into test and train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

#Applying all the classification algorithm on data for recognizing the voice whether the given frequancy of voice is of male and female
#Logistic regression

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)

y_pred = logmodel.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

#KKNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=5, 
                                      metric='minkowski', p=2)
classifier_knn.fit(x_train, y_train)
y_pred = classifier_knn.predict(x_test)
y_pred
confusion_matrix(y_test, y_pred)

#Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(x_train, y_train)
y_pred = classifier_nb.predict(x_test)

y_pred

confusion_matrix(y_test, y_pred)

#Support vector machine 
#SIGMOID MODE
from sklearn.svm import SVC
classifier_svm = SVC(kernel='sigmoid')
classifier_svm.fit(x_train,y_train)

y_pred = classifier_svm.predict(x_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

#LINEAR MODE
from sklearn.svm import SVC
classifier_svm = SVC(kernel='linear')
classifier_svm.fit(x_train,y_train)

y_pred = classifier_svm.predict(x_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

#RADIAL BASIS FORM
from sklearn.svm import SVC
classifier_svm = SVC(kernel='rbf')
classifier_svm.fit(x_train,y_train)

y_pred = classifier_svm.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

#POLYNOMIAL MODE
from sklearn.svm import SVC
classifier_svm = SVC(kernel='poly')
classifier_svm.fit(x_train,y_train)

y_pred = classifier_svm.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

#Decision Tree classifier

from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(criterion='entropy')
classifier_dt.fit(x_train,y_train)
y_pred = classifier_dt.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

#Random forest classifier

from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators=3, criterion='entropy')
classifier_rf.fit(x_train,y_train)

y_pred = classifier_rf.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


#The Logistic regression having great accuracy as compare to other it gives 97 percent accuracy. 

