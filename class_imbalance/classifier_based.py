import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from imblearn.datasets import make_imbalance
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from imblearn.metrics import geometric_mean_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import auc, precision_recall_curve, make_scorer
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier, EasyEnsembleClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, cross_validate

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
# remainder='passthrough': keeps the non transformed columns
minmax_transformer = ColumnTransformer(transformers=t, remainder='passthrough')  # use it on pipelines

print('\n============================================================================================================')


# ======================================================================================================================
# Set SCORING METRICS for cross_validate
# ======================================================================================================================

def precision_recall_auc_score(y_test_valid, y_positive_class_probs):
    # needs_proba=True from make_scorer keeps only the positive class probabilities
    precision, recall, _ = precision_recall_curve(y_test_valid, y_positive_class_probs)
    auc_score = auc(recall, precision)

    return auc_score


scoring = {'accuracy': 'accuracy', 'balanced-accuracy': 'balanced_accuracy', 'f1-score': 'f1_weighted', 'roc-auc': 'roc_auc',
           'precision-recall auc': make_scorer(precision_recall_auc_score, needs_proba=True, greater_is_better=True),
           'g-mean': make_scorer(geometric_mean_score, greater_is_better=True)}


# ======================================================================================================================
# LogisticRegression - COST-SENSITIVE LEARNING
# ======================================================================================================================

# Class imbalance unaware
clf1 = LogisticRegression(solver="liblinear", C=400, max_iter=300)
model1 = clf1.fit(X, y)

# Class imbalance aware
clf2 = LogisticRegression(solver="liblinear", C=400, max_iter=300, class_weight="balanced")
model2 = clf2.fit(X, y)

# ======================================================================================================================

predictors = [['Class imbalance unaware', clf1], ['Class imbalance aware', clf2]]

for name, classifier in predictors:
    steps = [('minmax', minmax_transformer), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nLogisticRegression - ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False, return_estimator=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

print('\n============================================================================================================')

# ======================================================================================================================

# Visualize DECISION BOUNDARY
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.Paired)
#print("HERE", plt.xlim(), ' AND ', plt.ylim())
#print('TEST ', plt.gca().get_ylim())
xx = np.linspace(plt.xlim(), 1000)

# plot the decision boundary of class imbalance UNAWARE
yy = (-model1.coef_[0][0]*xx-model1.intercept_[0])/model1.coef_[0][1]
plt.plot(xx, yy, "k-", label="default")

# plot the decision boundary of class imbalance AWARE
yy = (-model2.coef_[0][0]*xx-model2.intercept_[0])/model2.coef_[0][1]
plt.plot(xx, yy, "k--", label="aware of class imbalance")
plt.legend()

plt.show()


# ======================================================================================================================
# SVM - COST-SENSITIVE LEARNING
# ======================================================================================================================

# Class imbalance unaware
svc_imb_unaware = SVC(kernel='rbf', C=100, gamma='scale', probability=True)

# Class imbalance aware
svc_imb_aware = SVC(kernel='rbf', C=100, gamma='scale', class_weight='balanced', probability=True)

# ======================================================================================================================

predictors = [['Class imbalance unaware', svc_imb_unaware], ['Class imbalance aware', svc_imb_aware]]

for name, classifier in predictors:
    steps = [('minmax', minmax_transformer), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nSVM - ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False, return_estimator=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

print('\n============================================================================================================')


# ======================================================================================================================
# DecisionTreeClassifier - COST-SENSITIVE LEARNING
# ======================================================================================================================

# Class imbalance unaware
dtc_imb_unaware = DecisionTreeClassifier(criterion='gini', max_features='log2', max_depth=None, random_state=random_state)

# Class imbalance aware
dtc_imb_aware = DecisionTreeClassifier(criterion='gini', max_features='log2', max_depth=None, class_weight="balanced", random_state=random_state)

# ======================================================================================================================

predictors = [['Class imbalance unaware', dtc_imb_unaware], ['Class imbalance aware', dtc_imb_aware]]

for name, classifier in predictors:
    steps = [('minmax', minmax_transformer), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nDecisionTreeClassifier - ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False, return_estimator=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

print('\n============================================================================================================')


# ======================================================================================================================
# RandomForestClassifier - COST-SENSITIVE LEARNING
# ======================================================================================================================

# Class imbalance unaware
rf_imb_unaware = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=None,
                                        bootstrap=True, oob_score=False)

# Class imbalance aware
rf_imb_aware = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=None,
                                      bootstrap=True, oob_score=False, class_weight="balanced")

# Class imbalance aware
rf_imb_aware_2 = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=None,
                                        bootstrap=True, oob_score=False, class_weight='balanced_subsample')  # Bootstrap Class Weighting

# Class imbalance aware
rf_imb_aware_3 = BalancedRandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=None,
                                                bootstrap=True, oob_score=False)

# ======================================================================================================================

predictors = [['Class imbalance unaware', rf_imb_unaware], ['Balanced - Class imbalance aware', rf_imb_aware],
              ['Balanced subsample - Class imbalance aware', rf_imb_aware_2], ['Balanced Random Forest Classifier - Class imbalance aware', rf_imb_aware_3]]

for name, classifier in predictors:
    steps = [('minmax', minmax_transformer), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nRandomForestClassifier\n", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False, return_estimator=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

print('\n============================================================================================================')


# ======================================================================================================================
# XGBClassifier
# ======================================================================================================================

# Class imbalance unaware
xgb_imb_uaware = XGBClassifier(learning_rate=0.01, n_estimators=1000, max_depth=4, min_child_weight=6, gamma=0,
                               subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005, objective='binary:logistic',
                               nthread=4, scale_pos_weight=1,
                               random_state=random_state)

# Class imbalance aware
# scale_pos_weight = total_negative_examples / total_positive_examples
# scale_pos_weight: scale the gradient for the positive class, set to inverse of the class distribution (ratio 1:5 -> 5)
xgb_imb_aware = XGBClassifier(learning_rate=0.01, n_estimators=1000, max_depth=4, min_child_weight=6, gamma=0,
                              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005, objective='binary:logistic',
                              nthread=4, scale_pos_weight=int(y.value_counts()[0] / y.value_counts()[1]),
                              random_state=random_state)

# ======================================================================================================================

predictors = [['Class imbalance unaware', xgb_imb_uaware], ['Class imbalance aware', xgb_imb_aware]]

for name, classifier in predictors:
    steps = [('minmax', minmax_transformer), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nXGBClassifier - ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False, return_estimator=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

print('\n============================================================================================================')


# ======================================================================================================================
# Bagging
# ======================================================================================================================

# base_estimator: dtc_imb_unaware (used above) -> imbalance UNAWARE DecisionTreeClassifier
# bootstrap: whether samples are drawn with replacement
# bootstrap_features: whether features are drawn with replacement
# oob_score: whether to use out-of-bag samples to estimate the generalization error (need to be FALSE if warm_start TRUE)
# warm_start: when set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new ensemble.

# Class imbalance unaware
bagging_imb_unaware = BaggingClassifier(n_estimators=100, base_estimator=dtc_imb_aware,
                                        bootstrap_features=True, bootstrap=True, warm_start=False, oob_score=False,
                                        random_state=random_state)

# Class imbalance aware
bagging_imb_aware = BalancedBaggingClassifier(n_estimators=100, base_estimator=dtc_imb_aware,
                                              bootstrap_features=True, bootstrap=True, warm_start=False, oob_score=False,
                                              random_state=random_state)  # Bagging With Random Undersampling

# ======================================================================================================================

predictors = [['Class imbalance unaware', bagging_imb_unaware], ['Class imbalance aware', bagging_imb_aware]]

for name, classifier in predictors:
    steps = [('minmax', minmax_transformer), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\nBagging - ", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False, return_estimator=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

print('\n============================================================================================================')


# ======================================================================================================================
# EasyEnsembleClassifier
# ======================================================================================================================

# for AdaBoostClassifier -> base_estimator: dtc_imb_unaware (used above) -> imbalance UNAWARE DecisionTreeClassifier
eec = EasyEnsembleClassifier(n_estimators=10,
                             base_estimator=AdaBoostClassifier(base_estimator=None, n_estimators=10, algorithm='SAMME.R', random_state=random_state),
                             warm_start=False, random_state=random_state)

# ======================================================================================================================

steps = [('minmax', minmax_transformer), ('model', eec)]  # minmax scaler for continuous features
pipeline = Pipeline(steps=steps)

# evaluate pipeline
print("\nEasyEnsembleClassifier:")
scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False, return_estimator=False)
for s in scoring:
    print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

print('\n============================================================================================================')
