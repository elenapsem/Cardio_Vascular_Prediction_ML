import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib as mpl
from sklearn import tree
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from modAL.models import ActiveLearner
from sklearn.metrics import classification
from sklearn.naive_bayes import GaussianNB
from modAL.uncertainty import entropy_sampling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error,precision_recall_curve
from sklearn.metrics import auc, precision_recall_curve, make_scorer


instances = 1000
data = pd.read_csv('cardio_train.csv', sep=";")
seed = data.sample(instances)
X_raw = seed.drop('cardio', axis=1).values
y_raw = seed['cardio'].values


# Set our RNG seed for reproducibility.
RANDOM_STATE_SEED = 123
np.random.seed(RANDOM_STATE_SEED)
# Define our PCA transformer and fit it onto our raw dataset.
pca = PCA(n_components=2, random_state=RANDOM_STATE_SEED)
transformed_cardio = pca.fit_transform(X=X_raw)
# Isolate the data we'll need for plotting.
x_component, y_component = transformed_cardio[:, 0], transformed_cardio[:, 1]


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
learner = ActiveLearner(estimator=KNeighborsClassifier(n_neighbors=3), X_training=X_train, y_training=y_train)
# learner = ActiveLearner(estimator=RandomForestClassifier(), X_training=X_train, y_training=y_train)
# learner = ActiveLearner(estimator=GaussianNB(), X_training=X_train, y_training=y_train)
# learner = ActiveLearner(estimator=tree.DecisionTreeClassifier(criterion="gini", max_depth=3),
# X_training=X_train, y_training=y_train)
# learner = ActiveLearner(estimator=LogisticRegression(solver="liblinear", C=400, max_iter=300),X_training=X_train, y_training=y_train)

# Isolate the data we'll need for plotting.
predictions = learner.predict(X_raw)
is_correct = (predictions == y_raw)

# Record our learner's score on the raw data.
for i in range(len(y_raw)):
    y_predicted = learner.predict(X_raw)
    unqueried_score_accuracy = classification.balanced_accuracy_score(y_raw, y_predicted)
    unqueried_score_f1 = classification.f1_score(y_raw, y_predicted)
    # unqueried_score = roc_auc_score(y_raw, y_predicted)
    precision, recall, _ = precision_recall_curve(y_raw, y_predicted)
    unqueried_score_auc = auc(recall, precision)
    # unqueried_score = metrics.mean_squared_error(y_raw, y_predicted)
# print(unqueried_score_f1)

N_QUERIES = 200
performance_history_f1 = [unqueried_score_f1]
performance_history_accuracy = [unqueried_score_accuracy]
performance_history_auc = [unqueried_score_auc]


# Plot our classification results.
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
ax.scatter(x=x_component[is_correct],  y=y_component[is_correct],  c='g', marker='+', label='Correct',   alpha=8/10)
ax.scatter(x=x_component[~is_correct], y=y_component[~is_correct], c='r', marker='x', label='Incorrect', alpha=8/10)
ax.legend(loc='lower right')
ax.set_title("ActiveLearner class predictions (Accuracy: {score:.3f})".format(score=unqueried_score_accuracy))
plt.show()


# Allow our model to query our unlabeled dataset for the most
# informative points according to our query strategy (uncertainty sampling).
f1_list = []
max_f1 = 0
X_raw_list = []
y_raw_list = []
for index in range(N_QUERIES):
    query_index, query_instance = learner.query(X_pool)

    # Teach our ActiveLearner model the record it has requested.
    X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
    learner.teach(X=X, y=y)

    # Remove the queried instance from the unlabeled pool.
    X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

    # Calculate and report our model's accuracy.
    # model_accuracy = learner.score(X_raw, y_raw)
    y_predicted = learner.predict(X_raw)
    model_f1 = classification.f1_score(y_raw, y_predicted, average='weighted')
    model_accuracy = classification.balanced_accuracy_score(y_raw, y_predicted)
    # x, y, _ = precision_recall_curve(y_raw, y_predicted)
    # model_auc = auc(x, y)
    f1_list.append(model_f1)
    X_raw_list.append(X_raw)
    y_raw_list.append(y_raw)
    # print('F1 after query {n}: {f1:0.4f}'.format(n=index + 1, f1=model_f1))
    # print(X_raw, y_raw)
    # print(len(X_raw))
    # print(len(y_raw))
    # find max f1 and best x,y_raw
    if f1_list[index] > max_f1:
        max_f1 = f1_list[index]
        best_X_raw = X_raw_list[index]
        best_y_raw = y_raw_list[index]

    # Save our model's performance for plotting.
    performance_history_f1.append(model_f1)
    performance_history_accuracy.append(model_accuracy)
    # performance_history_auc.append(model_auc)

print(max_f1)
print(best_X_raw)
print(best_y_raw)

# Plot our performance over time.
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
ax.plot(performance_history_f1)
ax.scatter(range(len(performance_history_f1)), performance_history_f1, s=13)
ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_ylim(bottom=0, top=1)
ax.grid(True)
ax.set_title('Incremental classification f1')
ax.set_xlabel('Query iteration')
ax.set_ylabel('Classification F1')
plt.show()

fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
ax.plot(performance_history_accuracy)
ax.scatter(range(len(performance_history_accuracy)), performance_history_accuracy, s=13)
ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_ylim(bottom=0, top=1)
ax.grid(True)
ax.set_title('Incremental classification accuracy')
ax.set_xlabel('Query iteration')
ax.set_ylabel('Classification accuracy')
plt.show()
plt.savefig('bla.png')

# fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
# ax.plot(performance_history_auc)
# ax.scatter(range(len(performance_history_auc)), performance_history_auc, s=13)
# ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
# ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
# ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
# ax.set_ylim(bottom=0, top=1)
# ax.grid(True)
# ax.set_title('Incremental classification auc')
# ax.set_xlabel('Query iteration')
# ax.set_ylabel('Classification auc')
# plt.show()

# Isolate the data we'll need for plotting.
predictions = learner.predict(X_raw)
is_correct = (predictions == y_raw)

# Plot our updated classification results once we've trained our learner.
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
ax.scatter(x=x_component[is_correct],  y=y_component[is_correct],  c='g', marker='+', label='Correct',   alpha=8/10)
ax.scatter(x=x_component[~is_correct], y=y_component[~is_correct], c='r', marker='x', label='Incorrect', alpha=8/10)
ax.set_title('Classification accuracy after {n} queries: {final_acc:.3f}'.format(n=N_QUERIES, final_acc=performance_history_accuracy[-1]))
ax.legend(loc='lower right')
plt.show()