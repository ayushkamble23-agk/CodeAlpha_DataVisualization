import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

sns.set_style("darkgrid")

plt.figure(figsize=(8, 5))
sns.histplot(df['Fare'], bins=30, kde=True, color='teal')
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', data=df, palette='Set2')
plt.title("Passenger Class Count")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

df['Age'] = df['Age'].fillna(df['Age'].median())

plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=df, errorbar=None, palette='pastel')
plt.title("Survival Rate by Gender")
plt.ylabel("Proportion Survived")
plt.xlabel("Gender")
plt.show()

plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x='Age', hue='Survived', fill=True, common_norm=False, palette='crest', alpha=0.5)
plt.title("Age Distribution by Survival Status")
plt.xlabel("Age")
plt.ylabel("Density")
plt.show()

features_to_plot = ['Survived', 'Pclass', 'Age', 'Fare']
sns.pairplot(df[features_to_plot], hue='Survived', palette='bwr', diag_kind='kde')
plt.suptitle("Pairplot of Key Titanic Features", y=1.02)
plt.show()