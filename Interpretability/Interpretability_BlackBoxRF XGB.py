# Interpreatability for cardio vascular dataset
# Black Box model interpretation
# ======================================================================================================================
# Credits: https://colab.research.google.com/drive/1PuXCVSS4jjK3psMu7pwNCTlsmnIlVQ9T#scrollTo=CzY2OYJN9krH code

# installations, if not already there
#!pip install lime
#!pip install eli5
#!pip install shap
#!pip install pdpbox
#!pip install xgboost 

# imports
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost
from sklearn import datasets,model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, plot_importance
from IPython.display import SVG
from IPython.display import display
import matplotlib.pyplot as plt
from ipywidgets import interactive
from graphviz import Source

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import graphviz


# PDP
from pdpbox import pdp, get_dataset, info_plots

# LIME
import lime.lime_tabular
from lime.explanation import Explanation

# ELI5
import eli5
from eli5.sklearn import PermutationImportance

# SHAP
import shap

%matplotlib inline

# ======================================================================================================================
# read and preprocess data

df = pd.read_csv('cardio_train.csv', sep=";")
df.drop('id', 1, inplace=True)  # drop column 'id' as non relevant 
df['age'] = np.floor(df['age'] / 365.25)  # convert age

df.replace("", float("NaN"), inplace=True)  # convert empty field to NaN
df.dropna(inplace=True)  # drop NaN rows
df.reset_index(drop=True, inplace=True)

# extract features and target
X = df.iloc[:, :-1].values 
y = df.iloc[:, -1].values  


# assign features name
feature_names = df.columns.drop(['cardio'])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=0)


i = np.random.randint(0, x_test.shape[0]) # test for random instance

# ======================================================================================================================

# if a feature has 5 or less unique values then treat it as categorical
numerical_features= ["age", "height", "weight", "ap_hi", "ap_lo"]
# categorical_features = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
# if a feature has 5 or less unique values then treat it as categorical
categorical_features = np.argwhere(np.array([len(set(x_train[:,x])) for x in range(x_train.shape[1])]) <= 5).flatten()


# ====================================================================================================================== 
# Black Box Models interpretation
# ====================================================================================================================== 
#

# reuse the https://colab.research.google.com/drive/1PuXCVSS4jjK3psMu7pwNCTlsmnIlVQ9T#scrollTo=CzY2OYJN9krH code

def plot_tree(depth):
    estimator = DecisionTreeClassifier(random_state = 0,criterion = 'gini', max_depth = depth)
    estimator.fit(new_x_train, new_y_train)
    graph = Source(export_graphviz(estimator, out_file=None, feature_names=feature_names, filled = True))
    print("Fidelity",accuracy_score(y_pred, estimator.predict(x_test)))
    print("Accuracy in new data")
    print(accuracy_score(y_test, estimator.predict(x_test)))
    
    display(SVG(graph.pipe(format='svg')))
    
    return estimator


# Classifiers XGB / RandomForest

rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None)
xgb = XGBClassifier()

predictors = [ ['XGBClassifier', xgb], ['RandomForestClassifier', rfc]]

for name, classifier in predictors:
    
    classifier.fit(x_train, y_train)
    print(classifier)
          
    y_pred = classifier.predict(x_test)
    #print(accuracy_score(y_test,y_pred))    
    print(classification_report(y_test, y_pred))
    
    new_x_train = x_train
    new_y_train = classifier.predict(x_train)
    
    print("Decision Tree Explanator")
    
    inter=interactive(plot_tree,depth=(1,5))
    display(inter)
    
    # ======================================================================================================================
    # Different Approches deployed to check locally and gloabally

    # Local models
    
    # LIME
    explainer = lime.lime_tabular.LimeTabularExplainer(x_train, feature_names=feature_names, class_names=[0, 1],categorical_features=categorical_features, discretize_continuous=True)
    exp = explainer.explain_instance(x_test[i], classifier.predict_proba, num_features=5)
    exp.show_in_notebook(show_table=True, show_all=False)
    exp.as_pyplot_figure();
    
    
    # feature importance 
    
    # feature importance  built-in
    importances = classifier.feature_importances_
    # print(importances)
    indices = np.argsort(importances)[::-1]
    feature_indices = [ind for ind in indices[:len(importances)]]

    # Print the feature ranking
    # print("Feature ranking:")

    for f in range(len(importances)):
        print(f+1, feature_names[feature_indices[f]], importances[indices[f]])

    plt.figure(figsize=(15,5))
    plt.title("Feature importances")
    bars = plt.bar(range(len(importances)), importances[indices[:len(importances)]], align="center")
    ticks = plt.xticks(range(len(importances)),feature_names[feature_indices[:]])
    plt.show()

    #feature importance with ELI5
    perm = PermutationImportance(classifier).fit(x_test, y_test)
    eli5.show_weights(perm, feature_names = feature_names.tolist())
    display(eli5.show_weights(perm, feature_names = feature_names.tolist()))
    #eli5.show_prediction(classifier, x_test[i], show_feature_values=True, feature_names=feature_names.tolist())
    #display( eli5.show_prediction(classifier, x_test[i], show_feature_values=True, feature_names=feature_names.tolist()))
 
    
    if classifier == rfc:
        #feature importance with the partial dependence plot PDP 

        #feature cholesterol
        pdp_goals = pdp.pdp_isolate(model=classifier, dataset=df, model_features=feature_names, feature='cholesterol')
        display(pdp.pdp_plot(pdp_goals, 'cholesterol'))
        plt.show()

        #feature ap_hi
        pdp_goals = pdp.pdp_isolate(model=classifier, dataset=df, model_features=feature_names, feature='ap_hi')
        display(pdp.pdp_plot(pdp_goals, 'ap_hi'))
        plt.show()

        #feature active
        pdp_goals = pdp.pdp_isolate(model=classifier, dataset=df, model_features=feature_names, feature='active')
        display(pdp.pdp_plot(pdp_goals, 'active'))
        plt.show()
        
        
    # SHAP for evaluating variable importance
  
    data= shap.kmeans(x_train, 3) 
    explainer = shap.KernelExplainer(classifier.predict, data)
    shap_values = explainer.shap_values(x_train, nsamples=100)

    # show how each feature contributes to shifting the prediction from the base value to the output value of the model either by decreasing or increasing the probability of our class.
    shap.force_plot(explainer.expected_value, shap_values[i], x_test[i], feature_names=feature_names)
    plt.savefig('SHAP_feature_.png', bbox_inches="tight")
    shap.summary_plot(shap_values, x_train, show=False, feature_names=feature_names)
    plt.savefig('SHAP_Summary_.png', bbox_inches="tight")
