import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('dataset/cardio_train.csv', sep=";")

# ======================================================================================================================

# print(df.isnull().sum())   # there are no null values

# 1 = women, 2 = men
df['gender'] = df['gender'].map({1: 'women', 2: 'men'})

# ======================================================================================================================

# distribution of cardio vs age
sns.set_context("paper", font_scale=2, rc={"font.size": 20, "axes.titlesize": 25, "axes.labelsize": 20})
sns.catplot(kind='count', data=df, x='age', hue='cardio', order=df['age'].sort_values().unique())
plt.title('Variation of age for each cardio - target class')
plt.show()

# ======================================================================================================================

# barplot of age vs gender with hue = cardio
sns.catplot(kind='bar', data=df, y='age', x='gender', hue='cardio')
plt.title('Distribution of age vs gender with the cardio - target class')
plt.show()
