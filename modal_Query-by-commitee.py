# https://modal-python.readthedocs.io/en/latest/content/examples/query_by_committee.html
import numpy as np
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib as mpl
from copy import deepcopy
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import tree, metrics, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, mean_squared_error
from modAL.uncertainty import entropy_sampling
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner, Committee
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('cardio_train.csv', sep=";")

seed = data.sample(100)
# generate the pool
X_pool = deepcopy(seed.drop('cardio', axis=1).values)
y_pool = deepcopy(seed['cardio'].values)

# initializing Committee members
n_members = 3
learner_list = list()

for member_idx in range(n_members):
    # initial training data
    n_initial = 2
    train_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)
    X_train = X_pool[train_idx]
    y_train = y_pool[train_idx]

    # creating a reduced copy of the data with the known instances removed
    X_pool = np.delete(X_pool, train_idx, axis=0)
    y_pool = np.delete(y_pool, train_idx)

    # initializing learner
    learner = ActiveLearner(estimator=RandomForestClassifier(), X_training=X_train, y_training=y_train)
    # learner = ActiveLearner(estimator=KNeighborsClassifier(n_neighbors=2), X_training=X_train, y_training=y_train)
    # learner = ActiveLearner(estimator=tree.DecisionTreeClassifier(criterion="gini", max_depth=3),
    #                         X_training=X_train, y_training=y_train)
    # learner = ActiveLearner(estimator=GaussianNB(), X_training=X_train, y_training=y_train)
    # learner = ActiveLearner(estimator=LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial'),
    # X_training=X_train, y_training=y_train)
    learner_list.append(learner)

# assembling the committee
committee = Committee(learner_list=learner_list)
unqueried_score = committee.score(X_pool, y_pool)
print(unqueried_score)

for i in range(len(y_pool)):
    y_predicted = committee.predict(X_pool)
    unqueried_score_f1 = classification.accuracy_score(y_pool, y_predicted)
    # unqueried_score_f1 = classification.f1_score(y_pool, y_predicted)
    # unqueried_score_f1 = roc_auc_score(y_pool, y_predicted)
    # fpr, tpr, thresholds = metrics.roc_curve(y_pool, y_predicted)
    # unqueried_score_f1 = metrics.auc(fpr, tpr)
    # unqueried_score_f1 = metrics.mean_squared_error(y_pool, y_predicted)
print(unqueried_score_f1)

performance_history = [unqueried_score_f1]

# query by committee
n_queries = 20
for idx in range(n_queries):
    query_idx, query_instance = committee.query(X_pool)
    committee.teach(
        X=X_pool[query_idx].reshape(1, -1),
        y=y_pool[query_idx].reshape(1, )
    )
    performance_history.append(committee.score(X_pool, y_pool))
    # remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)
print(X_pool)
print(type(X_pool))
print(len(X_pool))
print(y_pool)

# Plot our performance over time.
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
ax.plot(performance_history)
ax.scatter(range(len(performance_history)), performance_history, s=13)
ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_ylim(bottom=0, top=1)
ax.grid(True)
ax.set_title('Incremental classification accuracy')
ax.set_xlabel('Query iteration')
ax.set_ylabel('Classification Accuracy')
plt.show()