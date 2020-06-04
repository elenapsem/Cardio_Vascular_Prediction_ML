import numpy as np
import pandas as pd
from imblearn.datasets import make_imbalance
from imblearn.metrics import geometric_mean_score
from numpy import mean
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.utils import resample
from sklearn.decomposition import PCA
from imblearn.under_sampling import TomekLinks
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import ClusterCentroids
from sklearn.metrics import roc_auc_score, confusion_matrix, make_scorer, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score


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

# Visualize BALANCED DATA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plot_2d_space(X_pca, y, 'Imbalanced dataset (2 PCA components)')


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

# Visualize IMBALANCED DATA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plot_2d_space(X_pca, y, 'Imbalanced dataset (2 PCA components)')


# ======================================================================================================================
# Scale data
# ======================================================================================================================

# Min-Max scaling data
scale_continuous_columns = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']  # continuous features

t = [('num', MinMaxScaler(), scale_continuous_columns)]
minmax_transformer = ColumnTransformer(transformers=t)  # use it on pipelines


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


# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=27)



# ======================================================================================================================
# RANDOM OVERSAMLING with replacement
# ======================================================================================================================

# upsample minority
decease_upsampled = resample(decease,
                             replace=True,  # sample with replacement
                             n_samples=len(no_decease),  # match number in majority class
                             random_state=27)  # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([no_decease, decease_upsampled])

print("RANDOM OVERSAMPLING: ", upsampled.cardio.value_counts())

# trying logistic regression again with the balanced dataset
y_train = upsampled.cardio
X_train = upsampled.drop('cardio', axis=1)


# ======================================================================================================================
# RANDOM UNDERSAMPLING
# ======================================================================================================================

# downsample majority
no_decease_downsampled = resample(no_decease,
                                 replace=False,  # sample without replacement
                                 n_samples=len(decease),  # match minority n
                                 random_state=27)  # reproducible results

# combine minority and downsampled majority
downsampled = pd.concat([no_decease_downsampled, decease])

print("RANDOM UNDERSAMPLING: ", downsampled.cardio.value_counts())

y_train = downsampled.cardio
X_train = downsampled.drop('cardio', axis=1)








# ======================================================================================================================
# TOMEK LINKS UNDERSAMPLING
# ======================================================================================================================

tl = TomekLinks(sampling_strategy='majority')
X_tl, y_tl, id_tl = tl.fit_sample(X_train_imbalanced, y_train_imbalanced)

print('Removed indexes:', id_tl)
print('DATA???:', X_tl, y_tl)

pca = PCA(n_components=2)
X_tl_pca = pca.fit_transform(X_tl)

plot_2d_space(X_tl_pca, y_tl, 'Tomek links under-sampling')


# ======================================================================================================================
# Cluster Centroids UNDERSAMPLING
# ======================================================================================================================

# {0: 10}: parameter ratio, 10 elements from majority class (0), and all minority class (1)
cc = ClusterCentroids(sampling_strategy={0: 10})
X_cc, y_cc = cc.fit_sample(X_train_imbalanced, y_train_imbalanced)

pca = PCA(n_components=2)
X_cc_pca = pca.fit_transform(X_cc)
plot_2d_space(X_cc_pca, y_cc, 'Cluster Centroids under-sampling')


# ======================================================================================================================
# COST-SENSITIVE LEARNING
# ======================================================================================================================

rf = RandomForestClassifier(class_weight="balanced")

fit_model = rf.fit(X_train_imbalanced, y_train_imbalanced)

# predict on test set
rf_pred = rf.predict(X_test)


# ======================================================================================================================

# Train model
clf_3 = SVC(kernel='linear',
            class_weight='balanced',  # penalize
            probability=True)

clf_3.fit(X_train_imbalanced, y_train_imbalanced)

# Predict on training set
pred_y_3 = clf_3.predict(X_test)

# What about AUROC?
prob_y_3 = clf_3.predict_proba(X_test)
prob_y_3 = [p[1] for p in prob_y_3]
print(roc_auc_score(y, prob_y_3))
# 0.5305236678


# ======================================================================================================================
# TRAIN ML MODEL - USE TREES for imbalanced data
# ======================================================================================================================

# train model
rfc = RandomForestClassifier(n_estimators=10)
fit_model = rfc.fit(X_train_imbalanced, y_train_imbalanced)

# predict on test set
rfc_pred = rfc.predict(X_test)


# ======================================================================================================================
# AUC-ROC (need proba)
# ======================================================================================================================

# Predict class probabilities
prob_y_2 = rfc.predict_proba(X_test)

# Keep only the positive class
prob_y_2 = [p[1] for p in prob_y_2]

print(roc_auc_score(y, prob_y_2))


# ======================================================================================================================
# SMOTE and Random Undersampling
# define pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# values to evaluate
k_values = [1, 2, 3, 4, 5, 6, 7]
for k in k_values:
    # ==================================================================================================================
    # transform the dataset
    over = BorderlineSMOTE()
    # ==================================================================================================================
    over = SVMSMOTE()
    # ==================================================================================================================
    oversample = ADASYN()
    # ==================================================================================================================
    # oversample the minority class to have 50 percent examples of the majority class
    over = SMOTE(sampling_strategy=0.5, k_neighbors=k, random_state=27)
    under = RandomUnderSampler(sampling_strategy=0.5)  # reduce the number of examples in the majority class by 50 percent
    steps = [('minmax', minmax_transformer), ('over', over), ('under', under)]  # minmax scaler for continuous features
    #steps = [('over', over), ('under', under), ('model', DecisionTreeClassifier())]
    pipeline = Pipeline(steps=steps)
    # transform the dataset
    X, y = pipeline.fit_resample(X_train_imbalanced, y_train_imbalanced)

    # ==================================================================================================================
    # define model
    model = DecisionTreeClassifier()
    # ==================================================================================================================
    model = XGBClassifier()
    # ==================================================================================================================

    # evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X_train_imbalanced, y_train_imbalanced, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('Mean ROC AUC: %.3f' % mean(scores))
    print('> k=%d, Mean ROC AUC: %.3f' % (k, mean(scores)))


    # ==================================================================================================================
    # Confusion Matrix
    # ==================================================================================================================

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
