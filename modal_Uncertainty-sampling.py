# https://modal-python.readthedocs.io/en/latest/content/examples/pool-based_sampling.html#
import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

instances = 100

data = pd.read_csv('cardio_train.csv', sep=";")
seed = data.sample(instances)
X_raw = seed.drop('cardio', axis=1).values
y_raw = seed['cardio'].values

# Isolate our examples for our labeled dataset.
n_labeled_examples = X_raw.shape[0]
training_indices = np.random.randint(low=0, high=n_labeled_examples + 1, size=3)
X_train = X_raw[training_indices]
y_train = y_raw[training_indices]

# Isolate the non-training examples we'll be querying.
X_pool = np.delete(X_raw, training_indices, axis=0)
y_pool = np.delete(y_raw, training_indices, axis=0)

# Specify our core estimator along with it's active learning model.
# query_strategy=random_sampling,entropy_sampling, margin_sampling
# learner = ActiveLearner(estimator=KNeighborsClassifier(n_neighbors=3), X_training=X_train, y_training=y_train)
# learner = ActiveLearner(estimator=RandomForestClassifier(), X_training=X_train, y_training=y_train)
# learner = ActiveLearner(estimator=GaussianNB(), X_training=X_train, y_training=y_train)
# learner = ActiveLearner(estimator=tree.DecisionTreeClassifier(criterion="gini", max_depth=3),
# X_training=X_train, y_training=y_train)
# learner = ActiveLearner(estimator=LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial'),
# X_training=X_train, y_training=y_train)
# Isolate the data we'll need for plotting.
predictions = learner.predict(X_raw)
is_correct = (predictions == y_raw)

# Record our learner's score on the raw data.
unqueried_score = learner.score(X_raw, y_raw)
N_QUERIES = 50
performance_history = [unqueried_score]

# Allow our model to query our unlabeled dataset for the most
# informative points according to our query strategy (uncertainty sampling).
for index in range(N_QUERIES):
    query_index, query_instance = learner.query(X_pool)

    # Teach our ActiveLearner model the record it has requested.
    X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
    learner.teach(X=X, y=y)

    # Remove the queried instance from the unlabeled pool.
    X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

    # Calculate and report our model's accuracy.
    model_accuracy = learner.score(X_raw, y_raw)
    # print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

    # Save our model's performance for plotting.
    performance_history.append(model_accuracy)

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
