# installations, if not already performed
#!pip install lime
#!pip install eli5
#!pip install shap
#!pip install pdpbox
#!pip install xgboost 

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import graphviz
import statistics
import collections
import numpy as np
import pandas as pd
import xgboost
from tqdm import tqdm
import seaborn as sns
from numpy import mean
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn import datasets,model_selection
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
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import SVG, display
from ipywidgets import interactive
from graphviz import Source
from pdpbox import pdp, get_dataset, info_plots # PDP
import lime.lime_tabular # LIME
from lime.explanation import Explanation
import eli5 # ELI5
from eli5.sklearn import PermutationImportance
import shap # SHAP
%matplotlib inline

# ======================================================================================================================

random_state = 27  # define random state for re-producability

# ======================================================================================================================

df = pd.read_csv('dataset/cardio_train.csv', sep=";")
df.drop('id', 1, inplace=True)  # delete column 'id'

#print(df.isnull().sum())   # there are no null values
df.replace("", float("NaN"), inplace=True)  # convert empty field to NaN
df.dropna(inplace=True)  # drop NaN rows
df.reset_index(drop=True, inplace=True)

df['age'] = np.floor(df['age'] / 365.25)  # convert age

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
# Black Box Model interpretation
# ====================================================================================================================== 
# reuse part of the https://colab.research.google.com/drive/1PuXCVSS4jjK3psMu7pwNCTlsmnIlVQ9T#scrollTo=CzY2OYJN9krH code
# ======================================================================================================================
# Plot Tree Function
# ======================================================================================================================
def plot_tree(depth):
    estimator = DecisionTreeClassifier(random_state = 0,criterion = 'gini', max_depth = depth)
    estimator.fit(new_x_train, new_y_train)
    graph = Source(export_graphviz(estimator, out_file=None, feature_names=feature_names,  class_names=labels, precision=7, filled = True))
    print("Fidelity",accuracy_score(y_pred, estimator.predict(x_test_scaled)))
    print("Accuracy in new data")
    print(accuracy_score(y_test, estimator.predict(x_test_scaled)))
    display(SVG(graph.pipe(format='svg')))
    return estimator


i = np.random.randint(0, x_test_scaled.shape[0]) # test for random instance

#under_over_x_train = under_over_x_train.values
#x_test = x_test.values
#under_over_y_train = under_over_y_train.values
y_test = y_test.values

# assign features names
feature_names = df.columns.drop(['cardio'])

numerical_features= ["age", "height", "weight", "ap_hi", "ap_lo"]
# categorical_features = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
# if a feature has 5 or less unique values then treat it as categorical
categorical_features = np.argwhere(np.array([len(set(under_over_x_train[:,x])) for x in range(under_over_x_train.shape[1])]) <= 5).flatten()

# Train as a black box with the preselected classifier
classifier= model_1 #classifier assignment

# Train decision tree as white box model to interpret the results 
new_x_train = under_over_x_train
new_y_train = classifier.predict(under_over_x_train)
    
print("Explain with Decision Tree")
inter=interactive(plot_tree,depth=(1,5))
display(inter)
    
# ======================================================================================================================
# Different Approches deployed to check results locally and globally
# ======================================================================================================================

# Local models  
# LIME
explainer = lime.lime_tabular.LimeTabularExplainer(under_over_x_train, feature_names=feature_names, class_names=[0, 1],categorical_features=categorical_features, discretize_continuous=True)
exp = explainer.explain_instance(x_test_scaled[i], classifier.predict_proba, num_features=5)
exp.show_in_notebook(show_table=True, show_all=False)
exp.as_pyplot_figure
    
# features importance 
    
# model built-in features importance
importances = classifier.feature_importances_
# print(importances)
indices = np.argsort(importances)[::-1]
feature_indices = [ind for ind in indices[:len(importances)]]

# Print the feature ranking
print("Feature ranking:")
for f in range(len(importances)):
    print(f+1, feature_names[feature_indices[f]], importances[indices[f]])

plt.figure(figsize=(15,5))
plt.title("Feature importances")
bars = plt.bar(range(len(importances)), importances[indices[:len(importances)]], align="center")
ticks = plt.xticks(range(len(importances)),feature_names[feature_indices[:]])
plt.show()

# features importance with ELI5
perm = PermutationImportance(classifier).fit(x_test_scaled, y_test)
eli5.show_weights(perm, feature_names = feature_names.tolist())
display(eli5.show_weights(perm, feature_names = feature_names.tolist()))
#eli5.show_prediction(classifier, x_test_scaled[i], show_feature_values=True, feature_names=feature_names.tolist())
#display( eli5.show_prediction(classifier, x_test_scaled[i], show_feature_values=True, feature_names=feature_names.tolist()))

# SHAP for evaluating variable importance
data= shap.kmeans(under_over_x_train, 3) 
explainer = shap.KernelExplainer(classifier.predict, data)
shap_values = explainer.shap_values(under_over_x_train, nsamples=100)

# show how each feature contributes to shifting the prediction from the base value to the output value of the model either by decreasing or increasing the probability of our class.
shap.force_plot(explainer.expected_value, shap_values[i], x_test_scaled[i], feature_names=feature_names)
plt.savefig('SHAP_feature_.png', bbox_inches="tight")
shap.summary_plot(shap_values, under_over_x_train, show=False, feature_names=feature_names)
plt.savefig('SHAP_Summary_.png', bbox_inches="tight")

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
# ====================================================================================================================== 
# Black Box Model interpretation
# ====================================================================================================================== 

# Train as a black box with the preselected classifier
classifier= model_2 #classifier assignment
  
new_x_train = under_over_x_train
new_y_train = classifier.predict(under_over_x_train)
    
print("Explain with Decision Tree")
inter=interactive(plot_tree,depth=(1,5))
display(inter)
    
# ======================================================================================================================
# Different Approches deployed to check results locally and gloabally
# ======================================================================================================================

# Local models
# LIME
explainer = lime.lime_tabular.LimeTabularExplainer(under_over_x_train, feature_names=feature_names, class_names=[0, 1],categorical_features=categorical_features, discretize_continuous=True)
exp = explainer.explain_instance(x_test_scaled[i], classifier.predict_proba, num_features=5)
exp.show_in_notebook(show_table=True, show_all=False)
exp.as_pyplot_figure();
    
# features importance 
    
#  model built-in features importance
importances = classifier.feature_importances_
# print(importances)
indices = np.argsort(importances)[::-1]
feature_indices = [ind for ind in indices[:len(importances)]]

# Print the feature ranking
print("Feature ranking:")
for f in range(len(importances)):
    print(f+1, feature_names[feature_indices[f]], importances[indices[f]])

plt.figure(figsize=(15,5))
plt.title("Feature importances")
bars = plt.bar(range(len(importances)), importances[indices[:len(importances)]], align="center")
ticks = plt.xticks(range(len(importances)),feature_names[feature_indices[:]])
plt.show()

# features importance with ELI5
perm = PermutationImportance(classifier).fit(x_test_scaled, y_test)
eli5.show_weights(perm, feature_names = feature_names.tolist())
display(eli5.show_weights(perm, feature_names = feature_names.tolist()))
#eli5.show_prediction(classifier, x_test_scaled[i], show_feature_values=True, feature_names=feature_names.tolist())
#display( eli5.show_prediction(classifier, x_test_scaled[i], show_feature_values=True, feature_names=feature_names.tolist()))

# SHAP for evaluating variable importance
data= shap.kmeans(under_over_x_train, 3) 
explainer = shap.KernelExplainer(classifier.predict, data)
shap_values = explainer.shap_values(under_over_x_train, nsamples=100)

# show how each feature contributes to shifting the prediction from the base value to the output value of the model either by decreasing or increasing the probability of our class.
shap.force_plot(explainer.expected_value, shap_values[i], x_test_scaled[i], feature_names=feature_names)
plt.savefig('SHAP_feature.png', bbox_inches="tight")
shap.summary_plot(shap_values, under_over_x_train, show=False, feature_names=feature_names)
plt.savefig('SHAP_Summary.png', bbox_inches="tight")
