import numpy as np
from pandas import read_csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
         'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
dataset = read_csv('adult.data', names=names)

array = dataset.to_numpy()
X = array

label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if isinstance(item, int | float):
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[::4, :-1].astype(int)
y = X_encoded[::4, -1].astype(int)


models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')), ('LDA', LinearDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier()), ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()),
          ('SVM', SVC(gamma='auto'))]

# noinspection DuplicatedCode
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_validate(model, X, y, cv=kfold, scoring=['f1', 'precision',  'accuracy', 'recall'])
    results.append(cv_results)
    names.append(name)
    print('------------------------')
    print(f'{name}')
    print(f'F1: {cv_results["test_f1"].mean():.2f}')
    print(f'Precision: {cv_results["test_precision"].mean():.2f}')
    print(f'Recall: {cv_results["test_recall"].mean():.2f}')
