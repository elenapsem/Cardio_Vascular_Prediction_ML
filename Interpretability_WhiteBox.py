# Interpreatability for cardio vascular dataset
# ======================================================================================================================
# Credits for Logistic Regression Model: https://colab.research.google.com/drive/1PuXCVSS4jjK3psMu7pwNCTlsmnIlVQ9T#scrollTo=CzY2OYJN9krH code

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
from sklearn import datasets,model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from IPython.display import SVG
from IPython.display import display
import matplotlib.pyplot as plt
from ipywidgets import interactive

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


# ======================================================================================================================

# if a feature has 5 or less unique values then treat it as categorical
numerical_features= ["age", "height", "weight", "ap_hi", "ap_lo"]
categorical_features = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
# if a feature has 5 or less unique values then treat it as categorical
categorical_features = np.argwhere(np.array([len(set(x_train[:,x])) for x in range(x_train.shape[1])]) <= 5).flatten()


# ====================================================================================================================== 
# White Box Models interpretation
# ====================================================================================================================== 

#LogisticRegression
# reuse the https://colab.research.google.com/drive/1PuXCVSS4jjK3psMu7pwNCTlsmnIlVQ9T#scrollTo=CzY2OYJN9krH code

model = LogisticRegression(solver="newton-cg",penalty='l2',max_iter=1000,C=100,random_state=0)
#lin_model = LogisticRegression(solver="liblinear",penalty='l1',max_iter=1000,C=10,random_state=0)
model.fit(x_train, y_train)
predicted_train = model.predict(x_train)
predicted_test = model.predict(x_test)
predicted_proba_test = model.predict_proba(x_test)
print("Logistic Regression Model Performance:")
print("Accuracy in Train Set",accuracy_score(y_train, predicted_train))
print("Accuracy in Test Set",accuracy_score(y_test, predicted_test))
              
# ======================================================================================================================

              
# ======================================================================================================================
# Features importance

weights = model.coef_
model_weights = pd.DataFrame({ 'features': list(feature_names),'weights': list(weights[0])})
#model_weights = model_weights.sort_values(by='weights', ascending=False) #Normal sort
model_weights = model_weights.reindex(model_weights['weights'].abs().sort_values(ascending=False).index) #Sort by absolute value
model_weights = model_weights[(model_weights["weights"] != 0)]    
print("Number of features:",len(model_weights.values))
plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
sns.barplot(x="weights", y="features", data=model_weights)
plt.title("Intercept (Bias): "+str(model.intercept_[0]),loc='right')
plt.xticks(rotation=90)
plt.show()


def plot_sensor(instance=0):
  random_instance = x_test[instance]
  print("Original Class:",y_test[instance],"Predicted Class:",predicted_test[instance],"with probability of", predicted_proba_test[instance][predicted_test[instance]])
  weights = model.coef_
  summation = sum(weights[0]*random_instance)
  bias = model.intercept_[0]
  res = ""
  if (summation + bias > 0):
    res = " > 0 -> 1"
  else:
    res = " <= 0 -> 0"
  print("Sum(weights*instance): "+str(summation)+" + Intercept (Bias): "+str(bias)+" = "+ str(summation+bias)+ res)
  model_weights = pd.DataFrame({ 'features': list(feature_names),'weights*values': list(weights[0]*random_instance)})
  #model_weights = model_weights.sort_values(by='weights*values', ascending=False)
  model_weights = model_weights.reindex(model_weights['weights*values'].abs().sort_values(ascending=False).index) #Sort by absolute value
  model_weights = model_weights[(model_weights["weights*values"] != 0)]    
  print("Number of features:",len(model_weights.values))
  plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
  sns.barplot(x="weights*values", y="features", data=model_weights)
  plt.xticks(rotation=90)
  plt.show()
              
inter=interactive(plot_sensor, instance=(0,9))
display(inter)
    

              
# ======================================================================================================================
# Different Approches

              
# Local models

#feature importance with ELI5
perm = PermutationImportance(model).fit(x_test, y_test)
eli5.show_weights(perm, feature_names = feature_names.tolist())
display(eli5.show_weights(perm, feature_names = feature_names.tolist()))

# LIME

i = np.random.randint(0, x_test.shape[0])
explainer = lime.lime_tabular.LimeTabularExplainer(x_train, feature_names=feature_names, class_names=[0, 1],categorical_features=categorical_features, discretize_continuous=True)
exp = explainer.explain_instance(x_test[i], model.predict_proba, num_features=5)
exp.show_in_notebook(show_table=True, show_all=False)
exp.as_pyplot_figure();

%matplotlib inline
fig = exp.as_pyplot_figure();


# Global models

#feature importance with the partial dependence plot PDP 

#feature cholesterol
pdp_goals = pdp.pdp_isolate(model=model, dataset=df, model_features=feature_names, feature='cholesterol')
display(pdp.pdp_plot(pdp_goals, 'cholesterol'))
plt.show()

#feature ap_hi
pdp_goals = pdp.pdp_isolate(model=model, dataset=df, model_features=feature_names, feature='ap_hi')
display(pdp.pdp_plot(pdp_goals, 'ap_hi'))
plt.show()


#feature active
pdp_goals = pdp.pdp_isolate(model=model, dataset=df, model_features=feature_names, feature='active')
display(pdp.pdp_plot(pdp_goals, 'active'))
plt.show()


# SHAP for evaluating variable importance


data= shap.kmeans(x_train, 3) 
explainer = shap.KernelExplainer(model.predict, data)
shap_values = explainer.shap_values(x_train, nsamples=100)

# show how each feature contributes to shifting the prediction from the base value to the output value of the model either by decreasing or increasing the probability of our class.
shap.force_plot(explainer.expected_value, shap_values[i], x_test[i], feature_names=feature_names)
plt.savefig('SHAPfeature_LR.png', bbox_inches="tight")
shap.summary_plot(shap_values, x_train, show=False, feature_names=feature_names)
plt.savefig('SHAPSummary_LR.png', bbox_inches="tight")
