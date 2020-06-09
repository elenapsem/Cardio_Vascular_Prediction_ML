import statistics
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from numpy import mean
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_validate, train_test_split
from imblearn.datasets import make_imbalance
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from imblearn.metrics import geometric_mean_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, RandomUnderSampler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN, RandomOverSampler
from sklearn.metrics import roc_auc_score, confusion_matrix, make_scorer, precision_recall_curve, auc, \
    balanced_accuracy_score, accuracy_score, precision_recall_fscore_support, f1_score


# ======================================================================================================================

random_state = 27  # define random state for re-producability

# ======================================================================================================================

df = pd.read_csv('dataset/cardio_train.csv', sep=";")
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
# INSERT ACTIVE LEARNING (instead of "Make the dataset IMBALANCED (with random undersampling)" below)
# ======================================================================================================================





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
# categorical_columns = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

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


# ======================================================================================================================
# Split to train and test data
# ======================================================================================================================

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=random_state)


# ======================================================================================================================
#  OVERSAMLING
# ======================================================================================================================

# RANDOM OVERSAMLING with replacement
random_over = RandomOverSampler(random_state=random_state)


# ======================================================================================================================
#  UNDERSAMPLING
# ======================================================================================================================

# cut half of the majority class (only for RandomUnderSampler)
under_sampling_strategy = {0: int(y_train.value_counts()[0] / 2), 1: y_train.value_counts()[1]}

# Tomek Links UNDERSAMPLING
tomek = TomekLinks(sampling_strategy='majority')

# RANDOM UNDERSAMPLING
random_under = RandomUnderSampler(sampling_strategy=under_sampling_strategy, random_state=random_state)  # reduce the number of examples in the majority class by 50 percent


# ======================================================================================================================
# CLASSIFIERS
# ======================================================================================================================

# LogisticRegression
clf2 = LogisticRegression(solver="liblinear", C=400, max_iter=300, class_weight="balanced")

# XGBClassifier
# scale_pos_weight: scale the gradient for the positive class, set to inverse of the class distribution (ratio 1:5 -> 5)
xgb_imb_aware = XGBClassifier(learning_rate=0.01, n_estimators=1000, max_depth=4, min_child_weight=6, gamma=0,
                            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005, objective='binary:logistic',
                            nthread=4, random_state=random_state)


# ======================================================================================================================
# MINMAX scaler for continuous features
# ======================================================================================================================

x_train_scaled = minmax_transformer.fit_transform(x_train, y_train)
x_test_scaled = minmax_transformer.transform(x_test)


# ======================================================================================================================
# TomekLinks + RandomOverSampler + XGBClassifier
# ======================================================================================================================

# UNDER-SAMPLING
# print(y_train.value_counts())
under_x_train, under_y_train = tomek.fit_resample(x_train_scaled, y_train)
# print(under_y_train.value_counts())


# OVER-SAMPLING
majority_class = under_y_train.value_counts()[0]
# over-sample 90% of the new difference of majority and minority labels (new = after under-sampling)
minority_class = under_y_train.value_counts()[1] + 0.80 * (under_y_train.value_counts()[0] - under_y_train.value_counts()[1])
over_sampling_strategy = {0: majority_class,
                          1: int(minority_class)}
random_over.set_params(sampling_strategy=over_sampling_strategy)
under_over_x_train, under_over_y_train = random_over.fit_resample(under_x_train, under_y_train)
# print(under_over_y_train.value_counts())


# set class imbalance parameter
xgb_imb_aware.set_params(scale_pos_weight=int(under_over_y_train.value_counts()[0] / under_over_y_train.value_counts()[1]))

# fit classifier
model_1 = xgb_imb_aware.fit(under_over_x_train, under_over_y_train)
y_pred = xgb_imb_aware.predict(x_test_scaled)
y_pred_proba = xgb_imb_aware.predict_proba(x_test_scaled)[:, 1]  # get only the probabilities for the positive class

print("\n", "TomekLinks + RandomOverSampler + XGBClassifier:")
print("accuracy:", accuracy_score(y_test, y_pred))
print("balanced accuracy: ", balanced_accuracy_score(y_test, y_pred))
print("weighted f1_score: ", f1_score(y_test, y_pred, average='weighted'))
print("roc-auc: ", roc_auc_score(y_test, y_pred_proba))
print("precision_recall_auc: ", precision_recall_auc_score(y_test, y_pred_proba))
print("g-mean: ", geometric_mean_score(y_test, y_pred, average='weighted'))

# ======================================================================================================================
# Confusion Matrix
# ======================================================================================================================

conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Confusion matrix:\n', conf_mat)

labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()


# ======================================================================================================================
# INSERT INTERPRETABILITY
# ======================================================================================================================









# ======================================================================================================================
# RandomUnderSampler + RandomOverSampler + XGBClassifier
# ======================================================================================================================

# UNDER-SAMPLING
# print(y_train.value_counts())
under_x_train, under_y_train = random_under.fit_resample(x_train_scaled, y_train)
# print(under_y_train.value_counts())


# OVER-SAMPLING
majority_class = under_y_train.value_counts()[0]
# over-sample 90% of the new difference of majority and minority labels (new = after under-sampling)
minority_class = under_y_train.value_counts()[1] + 0.80 * (under_y_train.value_counts()[0] - under_y_train.value_counts()[1])
over_sampling_strategy = {0: majority_class,
                          1: int(minority_class)}
random_over.set_params(sampling_strategy=over_sampling_strategy)
under_over_x_train, under_over_y_train = random_over.fit_resample(under_x_train, under_y_train)
# print(under_over_y_train.value_counts())


# set class imbalance parameter
xgb_imb_aware.set_params(scale_pos_weight=int(under_over_y_train.value_counts()[0] / under_over_y_train.value_counts()[1]))

# fit classifier
model_2 = xgb_imb_aware.fit(under_over_x_train, under_over_y_train)
y_pred = xgb_imb_aware.predict(x_test_scaled)
y_pred_proba = xgb_imb_aware.predict_proba(x_test_scaled)[:, 1]  # get only the probabilities for the positive class

print("\n", "RandomUnderSampler + RandomOverSampler + XGBClassifier:")
print("accuracy:", accuracy_score(y_test, y_pred))
print("balanced accuracy: ", balanced_accuracy_score(y_test, y_pred))
print("weighted f1_score: ", f1_score(y_test, y_pred, average='weighted'))
print("roc-auc: ", roc_auc_score(y_test, y_pred_proba))
print("precision_recall_auc: ", precision_recall_auc_score(y_test, y_pred_proba))
print("g-mean: ", geometric_mean_score(y_test, y_pred, average='weighted'))

# ======================================================================================================================
# Confusion Matrix
# ======================================================================================================================

conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Confusion matrix:\n', conf_mat)

labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()

# ======================================================================================================================
# INSERT INTERPRETABILITY
# ======================================================================================================================




