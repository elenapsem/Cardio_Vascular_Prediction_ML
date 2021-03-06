import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('dataset/cardio_train.csv', sep=";")

# ======================================================================================================================

# print(df.isnull().sum())   # there are no null values

# 1 = women, 2 = men
df['gender'] = df['gender'].map({1: 'women', 2: 'men'})

# age is given in days -> convert to years
df['age'] = round(df.age / 360)
df['age'] = df.age.astype(int)

print(df['age'])
# ======================================================================================================================
sns.boxplot(
    data=df,
    x='cardio',
    y='age',
    color='red')
plt.title('Age distribution per target label (cardio)')
plt.show()


sns.catplot(kind='count', data=df, x='age', col="cholesterol", hue='cardio', order=df['age'].sort_values().unique())
# plt.title('Distribution of age for each target class (cardio) per cholesterol levels')
plt.show()


sns.catplot(kind='count', data=df, x='age', col="active", hue='cardio', order=df['age'].sort_values().unique())
# plt.title('Distribution of age for each target class (cardio) per individual activity')
plt.show()

# distribution of cardio vs age
sns.countplot(x="age", hue="cardio", data=df, order=df['age'].sort_values().unique())
plt.title('Distribution of age for each target class (cardio)')
plt.show()

# ======================================================================================================================

# barplot of age vs gender with hue = cardio
sns.boxplot(
    data=df,
    x='gender',
    y='age',
    hue='cardio',
    color='red')
plt.title('Distribution of age vs gender with the target class (cardio)')
plt.show()
