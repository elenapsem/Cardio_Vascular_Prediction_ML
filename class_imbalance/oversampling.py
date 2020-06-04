import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from imblearn.datasets import make_imbalance
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, make_scorer
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN


# ======================================================================================================================

random_state = 27  # define random state for re-producability

# ======================================================================================================================

df = pd.read_csv('../dataset/cardio_train.csv', sep=";")
df.drop('id', 1, inplace=True)  # delete column 'id'

#print(df.isnull().sum())   # there are no null values
df.replace("", float("NaN"), inplace=True)  # convert empty field to NaN
df.dropna(inplace=True)  # drop NaN rows
df.reset_index(drop=True, inplace=True)

X = df.iloc[:, :-1]  # .values  # convert to numpy array
y = df.iloc[:, -1]  # .values  # convert to numpy array
# Separate input features and target
#y = df.cardio
#X = df.drop('cardio', axis=1)

#print(X, " and ", y)


# ======================================================================================================================
# VISUALIZE with PCA in 2D Graph
# ======================================================================================================================

def plot_2d_space(X, y, label='Classes'):
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y == l, 0],
            X[y == l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()


# ======================================================================================================================
'''
# Visualize BALANCED DATA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plot_2d_space(X_pca, y, 'Imbalanced dataset (2 PCA components)')
'''

# ======================================================================================================================
# Make the dataset IMBALANCED (with random undersampling)
# ======================================================================================================================

# downsample majority

print("BEFORE IMBALANCE: ", y.value_counts())

# keep all samples of class 0 and 1/5 samples of class 1
sampling_strategy = {0: y.value_counts()[0], 1: int(y.value_counts()[1] / 5)}
X, y = make_imbalance(X, y, sampling_strategy=sampling_strategy)

print("AFTER IMBALANCE: ", y.value_counts())


# ======================================================================================================================
'''
# Visualize IMBALANCED DATA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plot_2d_space(X_pca, y, 'Imbalanced dataset (2 PCA components)')
'''

# ======================================================================================================================
# Scale data
# ======================================================================================================================

# Min-Max scaling data
scale_continuous_columns = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']  # continuous features

t = [('num', MinMaxScaler(), scale_continuous_columns)]
minmax_transformer = ColumnTransformer(transformers=t)  # use it on pipelines

print('\n============================================================================================================')


# ======================================================================================================================
# Set SCORING METRICS for cross_validate
# ======================================================================================================================

def precision_recall_auc_score(y_test_valid, y_positive_class_probs):
    # needs_proba=True from make_scorer keeps only the positive class probabilities
    precision, recall, _ = precision_recall_curve(y_test_valid, y_positive_class_probs)
    auc_score = auc(recall, precision)

    return auc_score


scoring = {'accuracy': 'accuracy', 'balanced-accuracy': 'balanced_accuracy', 'f1-score': 'f1', 'roc-auc': 'roc_auc',
           'precision-recall auc': make_scorer(precision_recall_auc_score, needs_proba=True, greater_is_better=True),
           'g-mean': make_scorer(geometric_mean_score, greater_is_better=True)}


# ======================================================================================================================
# BASELINE MODEL (no balancing techniques employed)
# ======================================================================================================================

# LogisticRegression

lr = LogisticRegression(solver="liblinear", C=400, max_iter=200)

# GaussianNB
gnb = GaussianNB()

# RandomForestClassifier
# max_features: number of features that is randomly sampled for each split point
rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=None)

predictors = [['LogisticRegression', lr], ['GaussianNB', gnb], ['RandomForestClassifier', rfc]]

for name, classifier in predictors:
    steps = [('minmax', minmax_transformer), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nBaseline model + ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

print('\n============================================================================================================')


# ======================================================================================================================
# RANDOM OVERSAMLING with replacement
# ======================================================================================================================

over = RandomOverSampler(sampling_strategy='minority', random_state=random_state)

# ======================================================================================================================

# LogisticRegression
lr = LogisticRegression(solver="liblinear", C=400, max_iter=200)

# GaussianNB
gnb = GaussianNB()

# RandomForestClassifier
# max_features: number of features that is randomly sampled for each split point
rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=None)

predictors = [['LogisticRegression', lr], ['GaussianNB', gnb], ['RandomForestClassifier', rfc]]

for name, classifier in predictors:
    steps = [('minmax', minmax_transformer), ('over', over), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nRandom oversampling + ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

# check how many instances are removed in the whole dataset - only used as an indicator
_, y_under = over.fit_resample(X, y)
print('\nRandom - Final instances remained:\n', y_under.value_counts())

print('\n============================================================================================================')


# ======================================================================================================================
# SMOTE OVERSAMLING
# ======================================================================================================================

smote = SMOTE(k_neighbors=5, sampling_strategy='minority', random_state=random_state)

# ======================================================================================================================

# LogisticRegression
lr = LogisticRegression(solver="liblinear", C=400, max_iter=200)

# GaussianNB
gnb = GaussianNB()

# RandomForestClassifier
# max_features: number of features that is randomly sampled for each split point
rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=None)

predictors = [['LogisticRegression', lr], ['GaussianNB', gnb], ['RandomForestClassifier', rfc]]

for name, classifier in predictors:
    steps = [('minmax', minmax_transformer), ('over', smote), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nSMOTE oversampling + ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

# check how many instances are removed in the whole dataset - only used as an indicator
_, y_under = smote.fit_resample(X, y)
print('SMOTE - Final instances remained:\n', y_under.value_counts())

print('\n============================================================================================================')


# ======================================================================================================================
# BorderlineSMOTE OVERSAMLING
# ======================================================================================================================

border_smote = BorderlineSMOTE(k_neighbors=100, m_neighbors=50, sampling_strategy='minority', random_state=random_state)

# ======================================================================================================================

# LogisticRegression
lr = LogisticRegression(solver="liblinear", C=400, max_iter=400)

# GaussianNB
gnb = GaussianNB()

# RandomForestClassifier
# max_features: number of features that is randomly sampled for each split point
rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=None)

predictors = [['LogisticRegression', lr], ['GaussianNB', gnb], ['RandomForestClassifier', rfc]]

for name, classifier in predictors:
    steps = [('minmax', minmax_transformer), ('over', border_smote), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nBorderlineSMOTE oversampling + ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

# check how many instances are removed in the whole dataset - only used as an indicator
_, y_under = border_smote.fit_resample(X, y)
print('BorderlineSMOTE - Final instances remained:\n', y_under.value_counts())

print('\n============================================================================================================')


# ======================================================================================================================
# SVMSMOTE OVERSAMLING
# ======================================================================================================================

svm_smote = SVMSMOTE(k_neighbors=100, m_neighbors=100, svm_estimator=SVC(kernel='rbf', C=100, gamma='scale'), sampling_strategy='minority', random_state=random_state)

# ======================================================================================================================

# LogisticRegression
lr = LogisticRegression(solver="liblinear", C=400, max_iter=300)

# GaussianNB
gnb = GaussianNB()

# RandomForestClassifier
# max_features: number of features that is randomly sampled for each split point
rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=None)

predictors = [['LogisticRegression', lr], ['GaussianNB', gnb], ['RandomForestClassifier', rfc]]

for name, classifier in predictors:
    steps = [('minmax', minmax_transformer), ('over', svm_smote), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nSVMSMOTE oversampling + ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

# check how many instances are removed in the whole dataset - only used as an indicator
_, y_under = svm_smote.fit_resample(X, y)
print('SVMSMOTE - Final instances remained:\n', y_under.value_counts())

print('\n============================================================================================================')


# ======================================================================================================================
# ADASYN OVERSAMLING
# ======================================================================================================================

adasyn = ADASYN(n_neighbors=100, sampling_strategy='minority', random_state=random_state)


# ======================================================================================================================

# LogisticRegression
lr = LogisticRegression(solver="liblinear", C=400, max_iter=200)

# GaussianNB
gnb = GaussianNB()

# RandomForestClassifier
# max_features: number of features that is randomly sampled for each split point
rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=None)

predictors = [['LogisticRegression', lr], ['GaussianNB', gnb], ['RandomForestClassifier', rfc]]

for name, classifier in predictors:
    steps = [('minmax', minmax_transformer), ('over', adasyn), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nADASYN oversampling + ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

# check how many instances are removed in the whole dataset - only used as an indicator
_, y_under = adasyn.fit_resample(X, y)
print('ADASYN - Final instances remained:\n', y_under.value_counts())

print('\n============================================================================================================')
