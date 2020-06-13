import numpy as np
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib as mpl
from copy import deepcopy
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn import tree, metrics, svm
from sklearn.metrics import classification
from sklearn.naive_bayes import GaussianNB
from modAL.uncertainty import entropy_sampling
from modAL.models import ActiveLearner, Committee
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from modAL.disagreement import vote_entropy_sampling
from modAL.uncertainty import uncertainty_sampling
from sklearn.metrics import roc_auc_score, mean_squared_error


data = pd.read_csv('cardio_train.csv', sep=";")
seed = data.sample(1000)
# generate the pool
X_pool = deepcopy(seed.drop('cardio', axis=1).values)
y_pool = deepcopy(seed['cardio'].values)

# visualizing the classes
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    pca = PCA(n_components=2).fit_transform(seed.drop('cardio', axis=1))
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=seed['cardio'], cmap='viridis', s=50)
    plt.title('The cardio dataset')
    plt.show()

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
    print(X_pool)
    print(len(X_pool))
    print(y_pool)
    print(len(y_pool))

    # initializing learner
    learner = ActiveLearner(estimator=RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=None,
                                        bootstrap=True, oob_score=False), X_training=X_train, y_training=y_train)
    # learner = ActiveLearner(estimator=KNeighborsClassifier(n_neighbors=2), X_training=X_train, y_training=y_train)
    # learner = ActiveLearner(estimator=tree.DecisionTreeClassifier(criterion='gini', max_features='log2', max_depth=None),
                            # X_training=X_train, y_training=y_train)
    # learner = ActiveLearner(estimator=GaussianNB(), X_training=X_train, y_training=y_train)
    # learner = ActiveLearner(estimator=LogisticRegression(solver="liblinear", C=400, max_iter=300),
    # X_training=X_train, y_training=y_train)
    # learner = ActiveLearner(estimator=XGBClassifier(learning_rate=0.01, n_estimators=1000, max_depth=4, min_child_weight=6, gamma=0,
    #                                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005, objective='binary:logistic',
    #                                nthread=4, scale_pos_weight=1), X_training=X_train, y_training=y_train)
    learner_list.append(learner)
    print(learner_list)

print("committeeeee")
# assembling the committee
committee = Committee(learner_list=learner_list, query_strategy=uncertainty_sampling)
for i in range(len(y_pool)):
    y_predicted = committee.predict(X_pool)
    unqueried_score = classification.accuracy_score(y_pool, y_predicted)
    # unqueried_score = classification.f1_score(y_pool, y_predicted)
    # unqueried_score = roc_auc_score(y_pool, y_predicted)
    # fpr, tpr, thresholds = metrics.roc_curve(y_pool, y_predicted)
    # unqueried_score = metrics.auc(fpr, tpr)
    # unqueried_score = metrics.mean_squared_error(y_pool, y_predicted)
print(unqueried_score) #before queries


performance_history = [unqueried_score]

# query by committee
n_queries = 200
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
    print(len(X_pool))
    print(y_pool)
    print(len(y_pool))

# visualizing the Committee's predictions
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    prediction = committee.predict(seed.drop('cardio', axis=1))
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis', s=50)
    plt.title('Committee predictions after %d queries, accuracy = %1.3f'
              % (n_queries, committee.score(seed.drop('cardio', axis=1), seed['cardio'])))
    plt.show()

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