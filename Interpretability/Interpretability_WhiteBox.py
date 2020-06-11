# Interpreatability for cardio vascular dataset
# White Box model interpretation
# ======================================================================================================================
# Credits for Logistic Regression Model: https://colab.research.google.com/drive/1PuXCVSS4jjK3psMu7pwNCTlsmnIlVQ9T#scrollTo=CzY2OYJN9krH code

# installations, if not already there
#!pip install IPython
#!pip install eli5


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
import eli5
from eli5.sklearn import PermutationImportance


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
# White Box Models interpretation
# ====================================================================================================================== 

# LogisticRegression
# reuse the https://colab.research.google.com/drive/1PuXCVSS4jjK3psMu7pwNCTlsmnIlVQ9T#scrollTo=CzY2OYJN9krH code

model = LogisticRegression(solver="newton-cg",penalty='l2',max_iter=1000,C=100,random_state=0)
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
# ======================================================================================================================

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


def plot_sensor(instance):
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
              
inter=interactive(plot_sensor, instance=i)
display(inter)
    
              
# ======================================================================================================================
              
# features importance 

#features importance with ELI5
perm = PermutationImportance(model).fit(x_test, y_test)
eli5.show_weights(perm, feature_names = feature_names.tolist())
display(eli5.show_weights(perm, feature_names = feature_names.tolist()))

