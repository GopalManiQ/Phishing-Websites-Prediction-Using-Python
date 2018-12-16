#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing the dataset
dataset = pd.read_csv("Phishing Data.csv")
dataset.head()

#Splitting data and labels
x = dataset.iloc[ : , :-1].values
y = dataset.iloc[:, -1:].values

#Normalising values
from sklearn import preprocessing
normalized_x = preprocessing.normalize(x)

#spliting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(normalized_x,y,test_size = 0.25, random_state =0 )

#Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

logisticRegr = LogisticRegression()
SVMclassifier = SVC(kernel='linear')
KNNclassifier = KNeighborsClassifier(n_neighbors=5)
DecisionClassifier = DecisionTreeClassifier()
RFclassifier = RandomForestClassifier(n_estimators = 100, criterion = "gini", max_features = 'log2',  random_state = 0)


#Fitting Logistic Regression
logisticRegr.fit(x_train,y_train)

#Predicting the test result
y_pred_Logit = logisticRegr.predict(x_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

ConfusionMatrix_Logit = confusion_matrix(y_test, y_pred_Logit)
print(ConfusionMatrix_Logit)

ClassificationReport_Logit = classification_report(y_test, y_pred_Logit)
print(ClassificationReport_Logit)

#ROC Curve for logistic Regression
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_Logit)

#Fitting SVM
SVMclassifier.fit(x_train,y_train)

#Predicting the test result
y_pred_SVM = SVMclassifier.predict(x_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

ConfusionMatrix_SVM = confusion_matrix(y_test, y_pred_SVM)
print(ConfusionMatrix_SVM)

ClassificationReport_SVM = classification_report(y_test, y_pred_SVM)
print(ClassificationReport_SVM)

#Area under ROC Curve for SVM
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_SVM)


#Fitting Random Forest
RFclassifier.fit(x_train,y_train)

#Predicting the test result
y_pred_RF = RFclassifier.predict(x_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

ConfusionMatrix_RF = confusion_matrix(y_test, y_pred_RF)
print(ConfusionMatrix_RF)

ClassificationReport_RF = classification_report(y_test, y_pred_RF)
print(ClassificationReport_RF)

#Area under ROC Curve for random Forest
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_RF)

#Fitting Decision Tree model
DecisionClassifier.fit(x_train,y_train)

#Predicting the test result
y_pred_DT = DecisionClassifier.predict(x_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

ConfusionMatrix_DT = confusion_matrix(y_test, y_pred_DT)
print(ConfusionMatrix_DT)

ClassificationReport_DT = classification_report(y_test, y_pred_DT)
print(ClassificationReport_DT)

#Area under ROC Curve for Decision Tree
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_DT)

#Features Importance random forest
names = dataset.iloc[:,:-1].columns
importances = RFclassifier.feature_importances_
sorted_importances = sorted(importances, reverse=True)
indices = np.argsort(-importances)
var_imp = pd.DataFrame(sorted_importances, names[indices], columns=['importance'])


#plotting variable importance Random Forest
plt.title("Variable Importances")
plt.barh(np.arange(len(names)), sorted_importances, height = 0.7)
plt.yticks(np.arange(len(names)), names[indices], fontsize=7)
plt.xlabel('Relative Importance')
plt.show()


from sklearn.ensemble import GradientBoostingClassifier
GBClassifier = GradientBoostingClassifier(random_state=20)

#Fitting Gradient Boosting model
GBClassifier.fit(x_train,y_train)

y_pred_GB = GBClassifier.predict(x_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

ConfusionMatrix_GB = confusion_matrix(y_test, y_pred_GB)
print(ConfusionMatrix_GB)

ClassificationReport_GB = classification_report(y_test, y_pred_GB)
print(ClassificationReport_GB)

#Area under ROC Curve for Gradient Boosting
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_GB)


#ADA Boost
from sklearn.ensemble import AdaBoostClassifier
ADAClassifier = AdaBoostClassifier(n_estimators=10, random_state=7)

#Fitting ADA Boosting model
ADAClassifier.fit(x_train,y_train)
y_pred_ADA = ADAClassifier.predict(x_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

ConfusionMatrix_ADA = confusion_matrix(y_test, y_pred_ADA)
print(ConfusionMatrix_ADA)

ClassificationReport_ADA = classification_report(y_test, y_pred_ADA)
print(ClassificationReport_ADA)

#Area under ROC Curve for ADA Boost
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_ADA)

#Voting Classifier
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier

#Voting Ensemble
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
estimators.append(('logistic', logisticRegr))
estimators.append(('SVM', SVMclassifier))
estimators.append(('DecisionTree', DecisionClassifier))
estimators.append(('RandomForest', RFclassifier))

# create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(x_train,y_train)

y_pred_Ensemble = ensemble.predict(x_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

ConfusionMatrix_Ensemble = confusion_matrix(y_test, y_pred_Ensemble)
print(ConfusionMatrix_Ensemble)

ClassificationReport_Ensemble = classification_report(y_test, y_pred_Ensemble)
print(ClassificationReport_Ensemble)

#Area under ROC Curve for Voting Classifier
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_Ensemble)
