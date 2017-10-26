import os
import numpy as np
import timeit
from collections import defaultdict
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from genre import read_mfcc
genre_list = []

def model_training(X, Y, name):
    """
        Training and saving of model
    """
    clfs=[]
    cms=[]
    genre_labels = np.unique(Y)
    cv = ShuffleSplit(n=len(X), test_size=0.2, random_state=0)
    print("Logistic Regression: \n")
    clf = LogisticRegression()
    cv_scores = cross_val_score(clf, X, Y, cv=cv)
    print("Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
    Y_pred = cross_val_predict(clf,X,Y,cv=10)
    conf_mat = confusion_matrix(Y,Y_pred)
    print(conf_mat)
    cv = ShuffleSplit(n=len(X), test_size=0.2, random_state=0)
    print("SVM (rbf kernel) : \n")
    clf = svm.SVC(kernel='rbf')
    print("Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
    print(cv_scores)
    print(cv_scores.mean())
    y_pred = cross_val_predict(clf,X,Y,cv=10)
    conf_mat = confusion_matrix(Y,Y_pred)
    print(conf_mat)
    print("SVM (linear kernel) : \n")
    clf = svm.SVC(kernel='linear')
    print("Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
    print(cv_scores)
    print(cv_scores.mean())
    y_pred = cross_val_predict(clf,X,Y,cv=10)
    conf_mat = confusion_matrix(Y,Y_pred)
    print(conf_mat)
        
if __name__ == "__main__":
    start = timeit.default_timer()
    print("Classifier is running... Please wait. \n")
    X, y = read_mfcc()
    model_training(X, y, "first")

    print(" Classification finished \n")
