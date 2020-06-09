import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from imblearn.datasets import make_imbalance
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import auc, make_scorer, precision_recall_curve
from imblearn.under_sampling import NearMiss, CondensedNearestNeighbour, EditedNearestNeighbours, \
    RepeatedEditedNearestNeighbours, OneSidedSelection, NeighbourhoodCleaningRule, TomekLinks


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

def plot_2d_space(X_plot, y_plot, label='Classes'):
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y_plot), colors, markers):
        plt.scatter(
            X_plot[y_plot == l, 0],
            X_plot[y_plot == l, 1],
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

print("BEFORE IMBALANCE:\n", y.value_counts())

# keep all samples of class 0 and 1/5 samples of class 1
sampling_strategy = {0: y.value_counts()[0], 1: int(y.value_counts()[1] / 5)}
X, y = make_imbalance(X, y, sampling_strategy=sampling_strategy)

print("AFTER IMBALANCE:\n", y.value_counts())


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
# remainder='passthrough': keeps the non transformed columns
minmax_transformer = ColumnTransformer(transformers=t, remainder='passthrough')  # use it on pipelines


# ======================================================================================================================
# Set SCORING METRICS for cross_validate
# ======================================================================================================================

def precision_recall_auc_score(y_test_valid, y_positive_class_probs):
    # needs_proba=True from make_scorer keeps only the positive class probabilities
    precision, recall, _ = precision_recall_curve(y_test_valid, y_positive_class_probs)
    auc_score = auc(recall, precision)

    return auc_score


scoring = {'accuracy': 'accuracy', 'balanced-accuracy': 'balanced_accuracy', 'f1-score': 'f1_weighted', 'roc auc': 'roc_auc',
           'precision-recall auc': make_scorer(precision_recall_auc_score, needs_proba=True, greater_is_better=True),
           'g-mean': make_scorer(geometric_mean_score, greater_is_better=True)}

print('\n============================================================================================================')


# ======================================================================================================================
# BASELINE MODEL (no balancing techniques employed)
# ======================================================================================================================

# LogisticRegression
lr = LogisticRegression(solver="liblinear", C=500, max_iter=200)

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
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False, return_estimator=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

print('\n============================================================================================================')


# ======================================================================================================================
# RANDOM UNDERSAMPLING
# ======================================================================================================================

under = RandomUnderSampler(sampling_strategy='majority', random_state=random_state)  # reduce the number of examples in the majority class by 50 percent

# ======================================================================================================================

# LogisticRegression
lr = LogisticRegression(solver="liblinear", C=500, max_iter=200)

# GaussianNB
gnb = GaussianNB()

# RandomForestClassifier
# max_features: number of features that is randomly sampled for each split point
rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=None)

predictors = [['LogisticRegression', lr], ['GaussianNB', gnb], ['RandomForestClassifier', rfc]]

for name, classifier in predictors:
    steps = [('minmax', minmax_transformer), ('under', under), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nRandom undersampling + ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False, return_estimator=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

# check how many instances are removed in the whole dataset - only used as an indicator
_, y_under = under.fit_resample(X, y)
print('Random - Final instances remained:\n', y_under.value_counts())

print('\n============================================================================================================')


# ======================================================================================================================
# Tomek Links UNDERSAMPLING
# ======================================================================================================================

tomek = TomekLinks(sampling_strategy='majority')

# ======================================================================================================================

# LogisticRegression
# liblinear, sag, saga.
lr = LogisticRegression(solver="liblinear", C=500, max_iter=200)

# GaussianNB
gnb = GaussianNB()

# RandomForestClassifier
# max_features: number of features that is randomly sampled for each split point
rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=None)

predictors = [['LogisticRegression', lr], ['GaussianNB', gnb], ['RandomForestClassifier', rfc]]

for name, classifier in predictors:
    steps = [('minmax', minmax_transformer), ('under', tomek), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nTomek Links undersampling + ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False, return_estimator=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

# check how many instances are removed in the whole dataset - only used as an indicator
_, y_under = tomek.fit_resample(X, y)
print('Tomek Links - Final instances remained:\n', y_under.value_counts())

print('\n============================================================================================================')


# ======================================================================================================================
# NearMiss 1 UNDERSAMPLING
# ======================================================================================================================

nm1 = NearMiss(version=1, n_neighbors=3, sampling_strategy='majority')

# ======================================================================================================================

# LogisticRegression
# liblinear, sag, saga.
lr = LogisticRegression(solver="liblinear", C=500, max_iter=200)

# GaussianNB
gnb = GaussianNB()

# RandomForestClassifier
# max_features: number of features that is randomly sampled for each split point
rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=None)

predictors = [['LogisticRegression', lr], ['GaussianNB', gnb], ['RandomForestClassifier', rfc]]

for name, classifier in predictors:
    steps = [('minmax', minmax_transformer), ('under', nm1), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nNearMiss 1 undersampling + ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False, return_estimator=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

# check how many instances are removed in the whole dataset - only used as an indicator
_, y_under = nm1.fit_resample(X, y)
print('NearMiss 1 - Final instances remained:\n', y_under.value_counts())

print('\n============================================================================================================')


# ======================================================================================================================
# NearMiss 2 UNDERSAMPLING
# ======================================================================================================================

nm2 = NearMiss(version=2, n_neighbors=40, sampling_strategy='majority')

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
    steps = [('minmax', minmax_transformer), ('under', nm2), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nNearMiss 2 undersampling + ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False, return_estimator=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

# check how many instances are removed in the whole dataset - only used as an indicator
_, y_under = nm2.fit_resample(X, y)
print('NearMiss 2 - Final instances remained:\n', y_under.value_counts())

print('\n============================================================================================================')


# ======================================================================================================================
# NearMiss 3 UNDERSAMPLING
# ======================================================================================================================

nm3 = NearMiss(version=3, n_neighbors_ver3=3, sampling_strategy='majority')

# ======================================================================================================================

# LogisticRegression
#liblinear, sag, saga.
lr = LogisticRegression(solver="liblinear", C=500, max_iter=300)

# GaussianNB
gnb = GaussianNB()

# RandomForestClassifier
# max_features: number of features that is randomly sampled for each split point
rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=None) # gini

predictors = [['LogisticRegression', lr], ['GaussianNB', gnb], ['RandomForestClassifier', rfc]]

for name, classifier in predictors:
    steps = [('minmax', minmax_transformer), ('under', nm3), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nNearMiss 3 undersampling + ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False, return_estimator=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

# check how many instances are removed in the whole dataset - only used as an indicator
_, y_under = nm3.fit_resample(X, y)
print('NearMiss 3 - Final instances remained:\n', y_under.value_counts())

print('\n============================================================================================================')


# ======================================================================================================================
# EditedNearestNeighbours UNDERSAMPLING
# ======================================================================================================================

enn = EditedNearestNeighbours(n_neighbors=10, sampling_strategy='majority')

# ======================================================================================================================

# LogisticRegression
#liblinear, sag, saga.
lr = LogisticRegression(solver="liblinear", C=500, max_iter=200)

# GaussianNB
gnb = GaussianNB()

# RandomForestClassifier
# max_features: number of features that is randomly sampled for each split point
rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=None)

predictors = [['LogisticRegression', lr], ['GaussianNB', gnb], ['RandomForestClassifier', rfc]]

for name, classifier in predictors:
    steps = [('minmax', minmax_transformer), ('under', enn), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nEditedNearestNeighbours undersampling + ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False, return_estimator=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

# check how many instances are removed in the whole dataset - only used as an indicator
_, y_under = enn.fit_resample(X, y)
print('EditedNearestNeighbours - Final instances remained:\n', y_under.value_counts())

print('\n============================================================================================================')


# ======================================================================================================================
# RepeatedEditedNearestNeighbours UNDERSAMPLING
# ======================================================================================================================

renn = RepeatedEditedNearestNeighbours(n_neighbors=3, sampling_strategy='majority')

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
    steps = [('minmax', minmax_transformer), ('under', renn), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nRepeatedEditedNearestNeighbours undersampling + ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False, return_estimator=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

# check how many instances are removed in the whole dataset - only used as an indicator
_, y_under = renn.fit_resample(X, y)
print('RepeatedEditedNearestNeighbours - Final instances remained:\n', y_under.value_counts())

print('\n============================================================================================================')


# ======================================================================================================================
# OneSidedSelection UNDERSAMPLING
# ======================================================================================================================

# Remove borderline / noisy examples, defined by classifying them with 1-NN, that is trained base on a bucket that
# contains all minority class examples and n_seeds_S number of examples of the majority class. If the classification
# results do not agree with the true label then place the instances of the majority class in the bucket. When the
# process is over, remove all the instances of the majority class that were missclassified and are in the bucket.

# n_seeds_S: the minimum number of majority class instances to keep (default: 1)
oss = OneSidedSelection(n_neighbors=1, n_seeds_S=20, sampling_strategy='majority', random_state=random_state)

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
    steps = [('minmax', minmax_transformer), ('under', oss), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nOneSidedSelection undersampling + ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False, return_estimator=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

# check how many instances are removed in the whole dataset - only used as an indicator
_, y_under = oss.fit_resample(X, y)
print('OneSidedSelection - Final instances remained:\n', y_under.value_counts())

print('\n============================================================================================================')


# ======================================================================================================================
# NeighbourhoodCleaningRule UNDERSAMPLING
# ======================================================================================================================

ncr = NeighbourhoodCleaningRule(n_neighbors=10, threshold_cleaning=0.5, sampling_strategy='majority')

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
    steps = [('minmax', minmax_transformer), ('under', ncr), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nNeighbourhoodCleaningRule undersampling + ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False, return_estimator=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

# check how many instances are removed in the whole dataset - only used as an indicator
_, y_under = ncr.fit_resample(X, y)
print('NeighbourhoodCleaningRule - Final instances remained:\n', y_under.value_counts())

print('\n============================================================================================================')


# ======================================================================================================================
# Cluster Centroids UNDERSAMPLING
# ======================================================================================================================

cc = ClusterCentroids(sampling_strategy='majority', random_state=random_state)

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
    steps = [('minmax', minmax_transformer), ('under', cc), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nCluster Centroids undersampling + ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=2, return_train_score=False, return_estimator=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

# check how many instances are removed in the whole dataset - only used as an indicator
_, y_under = cc.fit_resample(X, y)
print('Cluster Centroids - Final instances remained:\n', y_under.value_counts())

print('\n============================================================================================================')


# ======================================================================================================================
# CondensedNearestNeighbour UNDERSAMPLING
# ======================================================================================================================

cnn = CondensedNearestNeighbour(n_neighbors=1, sampling_strategy='majority', random_state=random_state)

# ======================================================================================================================

# LogisticRegression
lr = LogisticRegression(solver="liblinear", C=500, max_iter=200)

# GaussianNB
gnb = GaussianNB()

# RandomForestClassifier
# max_features: number of features that is randomly sampled for each split point
rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=None)

predictors = [['LogisticRegression', lr], ['GaussianNB', gnb], ['RandomForestClassifier', rfc]]

for name, classifier in predictors:
    steps = [('minmax', minmax_transformer), ('under', cnn), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nCondensedNearestNeighbour undersampling + ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=2, return_train_score=False, return_estimator=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

# check how many instances are removed in the whole dataset - only used as an indicator
_, y_under = cnn.fit_resample(X, y)
print('CondensedNearestNeighbour - Final instances remained:\n', y_under.value_counts())

print('\n============================================================================================================')

