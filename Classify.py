from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import average_precision_score
import cPickle
import numpy as np
import itertools
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

rng = np.random.RandomState(1234)

#load pickle
positive = cPickle.load(open('OK_dataset', "rb"))
positive = np.array(positive)
negative = cPickle.load(open('WB_dataset', "rb"))
negative = np.array(negative)

#Concatinate"OK_dataset" and "WB_dataset"
dataset = np.vstack((positive, negative))
number_of_samples = dataset.shape[0]

# Labeling
labels = np.zeros((number_of_samples))
labels[positive.shape[0]+1:]=1

# Shuffling
rd = rng.permutation(number_of_samples)

    # X_valus or features
dataset = dataset[rd]
    # Y_values or labels
labels = labels[rd]

X = dataset #Feature Set
y = labels   #Label Set
print "Dataset_size = ",X.shape

#Feature selection based on mutual information

sel = SelectKBest(mutual_info_classif, k=3)
sel.fit(X, y)
X = sel.transform(X)


#Apply Stratification for Cross validation

print "\nTrain and tast Folds"

skf = StratifiedKFold(n_splits=3)
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# Ensemble Methods
''' 1. Random Forests,
    2. AdaBoost,
    3. Gradient Tree Boosting,
    4. Voting Classifier.'''


print "\n==============================="
print "      VOTING CLASSIFIER"
print "===============================\n"

#=========================
'''Voting Classifier'''
#=========================
'''Start measuring computation time'''
start = time.time()

clf =  VotingClassifier(
    estimators=[('lr', LogisticRegression()),
                ('rf', RandomForestClassifier()),
                ('gnb', GaussianNB())],
    voting='hard')

        # Fit Voting Classifier
y_pred = clf.fit(X_train, y_train).predict(X_test)

end = time.time()

print "Computation time of Voting Classifier =",(end - start),"Seconds"

print "prediction accuracy of Voting Classifier =",clf.score(X, y)
print "mean_squared_error of Voting Classifier =",mean_squared_error(y_test, y_pred)

    #Generate classification report
target_names = ['y_test', 'y_pred']
print "\nCLASSIFICATION REPORT FOR VOTING CLASSIFIER"
print(classification_report(y_test, y_pred, target_names=target_names))


           #CONFUSION MATRIX
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\n NORMALIZED CONFUSION MATRIX")
    else:
        print('\n ONFUSION MATRIX WITHOUT NORMALIZATION')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
#plt.figure()
plot_confusion_matrix(cnf_matrix, classes=y,
                      title='CONFUSION MATRIX WITHOUT NORMALIZATION')

# Plot normalized confusion matrix
#plt.figure()
plot_confusion_matrix(cnf_matrix, y, normalize=True,
                      title='NORMALIZED CONFUSION MATRIX')

#plt.show()

print "\n========================================"
print "      GRADIENT TREE BOOSTING CLASSIFIER"
print "==========================================\n"


#=======================================
'''Gradient Tree Boosting Classifier'''
#=======================================

clf = GradientBoostingClassifier(n_estimators=12)

 # Fit Voting Classifier
y_pred = clf.fit(X_train, y_train).predict(X_test)

end = time.time()

print "Computation time of Gradient Tree Boosting Classifier =",(end - start),"Seconds"

print "prediction accuracy of Gradient Tree Boosting Classifier =",clf.score(X, y)
print "mean_squared_error of Gradient Tree Boosting Classifier =",mean_squared_error(y_test, y_pred)

    #Generate classification report
target_names = ['y_test', 'y_pred']
print "\nCLASSIFICATION REPORT FOR GRADIENT TREE BOOSTING CLASSIFIER"
print(classification_report(y_test, y_pred, target_names=target_names))


           #CONFUSION MATRIX
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\n NORMALIZED CONFUSION MATRIX")
    else:
        print('\n CONFUSION MATRIX WITHOUT NORMALIZATION')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
#plt.figure()
plot_confusion_matrix(cnf_matrix, classes=y,
                      title='CONFUSION MATRIX WITHOUT NORMALIZATION')

# Plot normalized confusion matrix
#plt.figure()
plot_confusion_matrix(cnf_matrix, y, normalize=True,
                      title='NORMALIZED CONFUSION MATRIX')

#plt.show()



print "\n========================================"
print "      ADABOOST CLASSIFIER"
print "=========================================\n"


#=======================================
'''AdaBoost'''
#=======================================

clf = AdaBoostClassifier(base_estimator=None, n_estimators=100)



 # Fit Voting Classifier
y_pred = clf.fit(X_train, y_train).predict(X_test)

end = time.time()

print "Computation time of AdaBoost Classifier =",(end - start),"Seconds"

print "prediction accuracy of AdaBoost Classifier =",clf.score(X, y)
print "mean_squared_error of AdaBoost Classifier =",mean_squared_error(y_test, y_pred)

    #Generate classification report
target_names = ['y_test', 'y_pred']
print "\nCLASSIFICATION REPORT FOR ADABOOST CLASSIFIER"
print(classification_report(y_test, y_pred, target_names=target_names))


           #CONFUSION MATRIX
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\n NORMALIZED CONFUSION MATRIX")
    else:
        print('\n CONFUSION MATRIX WITHOUT NORMALIZATION')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
#plt.figure()
plot_confusion_matrix(cnf_matrix, classes=y,
                      title='CONFUSION MATRIX WITHOUT NORMALIZATION')

# Plot normalized confusion matrix
#plt.figure()
plot_confusion_matrix(cnf_matrix, y, normalize=True,
                      title='NORMALIZED CONFUSION MATRIX')

#plt.show()


print "\n========================================"
print "      RANDOM FOREST CLASSIFIER"
print "=========================================\n"


#=======================================
'''Random forest'''
#=======================================

clf = RandomForestClassifier(n_estimators=100, oob_score=True)



 # Fit Voting Classifier
y_pred = clf.fit(X_train, y_train).predict(X_test)

end = time.time()

print "Computation time of Random forest Classifier =",(end - start),"Seconds"

print "prediction accuracy of Random forest Classifier =",clf.score(X, y)
print "mean_squared_error of Random forest Classifier =",mean_squared_error(y_test, y_pred)

    #Generate classification report
target_names = ['y_test', 'y_pred']
print "\nCLASSIFICATION REPORT FOR RANDOM FOREST CLASSIFIER"
print(classification_report(y_test, y_pred, target_names=target_names))


           #CONFUSION MATRIX
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\n NORMALIZED CONFUSION MATRIX")
    else:
        print('\n CONFUSION MATRIX WITHOUT NORMALIZATION')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
#plt.figure()
plot_confusion_matrix(cnf_matrix, classes=y,
                      title='CONFUSION MATRIX WITHOUT NORMALIZATION')

# Plot normalized confusion matrix
#plt.figure()
plot_confusion_matrix(cnf_matrix, y, normalize=True,
                      title='NORMALIZED CONFUSION MATRIX')

#plt.show()
