"""
Titanic EDA Project 🚢📊

This project performs an exploratory data analysis (EDA) on the Titanic dataset.
It includes:
- Data cleaning (handling missing values, removing duplicates and unused columns)
- Age grouping into logical life stages
- Visualization of survival distribution.

All visuals use Seaborn styling with pastel palettes for clarity and elegance.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set seaborn style
sns.set(style="whitegrid")

# Load data
df = pd.read_csv('Titanic.csv')

# Clean data
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.fillna("Unknown", inplace=True)
df.drop_duplicates(inplace=True)
df.drop(columns=[col for col in ['Parch', 'Fare', 'SibSp', 'Ticket', 'Cabin'] if col in df.columns], inplace=True)
df['Survived'] = df['Survived'].map({0: 'Dead', 1: 'Survived'})

# Age grouping
age_bins = [0, 12, 20, 40, 60, 80]
age_labels = ['Child', 'Teenager', 'Adult', 'Middle Aged', 'Senior']
df['Agegroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

# Plot helpers
def pie_plot(series, title):
    plt.figure(figsize=(6, 6))
    counts = series.value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
    plt.title(title)
    plt.legend(title=series.name, loc="upper left")
    plt.show()

def count_plot(series, title):
    plt.figure(figsize=(6, 6))
    sns.countplot(x=series, palette="pastel")
    plt.title(title)
    plt.xlabel(series.name)
    plt.ylabel("Count")
    plt.show()

# Filter survived passengers
df_survived = df[df['Survived'] == 'Survived']

# Plots
count_plot(df['Survived'], "Survival Count")
pie_plot(df_survived['Sex'], "Survival by Gender")
pie_plot(df_survived['Agegroup'], "Survival by Age Group")

