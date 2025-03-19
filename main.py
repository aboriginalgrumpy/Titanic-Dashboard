from multiprocessing.connection import families

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


pd.set_option('display.max_column', None)
df = pd.read_csv("Titanic-Dataset2.csv")
df.dropna(subset=['Embarked'], inplace=True)

st.set_page_config(page_title="Titanic Dashboard", layout="wide")

st.title("Titanic Survival Assignment Dashboard")

st.sidebar.header("Filter Options")
selected_class = st.sidebar.multiselect("Select Passenger Class", df["Pclass"].unique(), default=df["Pclass"].unique())
selected_gender = st.sidebar.multiselect("Select Gender", df['Sex'].unique(), default=df['Sex'].unique())
selected_agegroup = st.sidebar.multiselect("Select Age Group", df['AgeGroup'].unique(), default=df['AgeGroup'].unique())

filtered_df = df[(df["Pclass"].isin(selected_class)) & (df["Sex"].isin(selected_gender)) & (df["AgeGroup"].isin(selected_agegroup))]

st.subheader("Survival Proportion")
survival_counts = filtered_df["Survived"].value_counts()
fig, ax=plt.subplots(figsize=(6,6))
ax.pie(survival_counts, labels=["Did not Survive", "Survived"], autopct="%1.1f%%", colors=["red", "green"])
st.pyplot(fig)

st.subheader("Survival Rate by Class & Gender")
fig, ax=plt.subplots(figsize=(8,5))
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=filtered_df, palette="coolwarm", ax=ax)
st.pyplot(fig)

st.subheader("Survival Rate by Family Size")
fig, ax=plt.subplots(figsize=(8,5))
sns.barplot(x="Family Size", y="Survived", hue="Family Size", data=filtered_df, palette="coolwarm", ax=ax)
st.pyplot(fig)

st.subheader("Survival Trends Across AgeGroups")
fig, ax=plt.subplots(figsize=(8,5))
sns.lineplot(x="AgeGroup", y="Survived", data=filtered_df, ax=ax)
st.pyplot(fig)



st.subheader("Correlation Between Fare Price & Survival")
fig, ax=plt.subplots(figsize=(8,5))
sns.heatmap(df.pivot_table(index="Pclass", columns="Survived", values="Fare"), cmap="coolwarm", annot=True, ax=ax)
st.pyplot(fig)

# st.subheader("Titanic Survival Correlation")
# corr_matrix = df.corr()
# fig, ax=plt.subplots(figsize=(10,6))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
# st.pyplot(fig)

df_corr = df.copy()
df_corr_numeric = df_corr.select_dtypes(include=["number"])

# correlation analysis using heatmap
correlation_matrix = df_corr_numeric.corr()
fig, ax=plt.subplots(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
st.subheader("Titanic survival correlation")
st.pyplot(fig)

df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very high'])

plt.savefig("dashboard.png", dpi=300, bbox_inches='tight')

# st.markdown("### Dashboard Impact:")
# st.markdown("- Dashboard helps to **improve interpretability** where the uses can visual and analyse key survival factors.")
# st.markdown("- Dashboard supports **decision-making** where the data driven insights can give insights for better evacuation plans")
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())


# print(df[['Age', 'AgeGroup', 'Fare', 'FareGroup', 'Embarked', 'Survived']].head())

# #group by FareGroup to get survival rate
# fare_survival = df.groupby('FareGroup')['Survived'].mean()
# print("Survival Rate by Fare Group")
# print(fare_survival)
#
# #group by AgeGroup to get survival rate
# age_survival = df.groupby('AgeGroup')['Survived'].mean()
# print("Survival Rate by Age Group")
# print(age_survival)
#
# #group by Embarked to get survival rate
# embarked_survival = df.groupby('Embarked')['Survived'].mean()
# print("Survival Rate by Embarked")
# print(embarked_survival)
#
# #matplotlib visual
# fig, axes = plt.subplots(1,3, figsize=(18,5))
#
# sns.barplot(x=age_survival.index, y=age_survival.values,palette="coolwarm", ax=axes[0])
# axes[0].set_title("Survival Rate by Age Group")
# axes[0].set_ylabel("Survival Probability")
# axes[0].set_xlabel("AgeGroup")
#
# sns.barplot(x=fare_survival.index, y=fare_survival.values, palette="coolwarm", ax=axes[1])
# axes[1].set_title("Survival Rate by Fare Group")
# axes[1].set_ylabel("Survival Probability")
# axes[1].set_xlabel("FareGroup")
#
# sns.barplot(x=embarked_survival.index, y=embarked_survival.values, palette="coolwarm", ax=axes[2])
# axes[2].set_title("Survival Rate by Embarked")
# axes[2].set_ylabel("Survival Prbability")
# axes[2].set_xlabel("Embarkation Port")
#
# plt.tight_layout()
# plt.show()

# sns.set_style("whitegrid")
#
# plt.figure(figsize=(8,5))
# sns.barplot(x=age_survival.index, y=age_survival.values, palette="coolwarm")
# plt.title("Survival Rate by AgeGroup")
# plt.ylabel("Survival probability")
# plt.show()
#
# plt.figure(figsize=(8,5))
# sns.barplot(x=fare_survival.index, y=fare_survival.values, palette="coolwarm")
# plt.title("Survival Rate by FareGroup")
# plt.ylabel("Survival Probability")
# plt.show()
#
# plt.figure(figsize=(8,5))
# sns.barplot(x=embarked_survival.index, y=embarked_survival.values, palette="coolwarm")
# plt.title("Survival Rate by Embarked port")
# plt.ylabel("Survival probability")
# plt.show()

#calculate Mean & Median for survival rates
# print("Mean:", df["Survived"].mean())
# print("Median:", df["Survived"].median())
#
# df_corr = df.copy()
#
# df_corr_numeric = df_corr.select_dtypes(include=["number"])

#correlation analysis using heatmap
# correlation_matrix = df_corr_numeric.corr()
# plt.figure(figsize=(8,6))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# plt.title("Titanic survival correlation")
# plt.show()
#
# correlation_matrix
#calculate survival rate
# survival_by_class = df.groupby("Pclass")["Survived"].mean()*100
# survival_by_gender = df.groupby("Sex")["Survived"].mean()*100
# survival_by_agegroup = df.groupby("AgeGroup")["Survived"].mean()*100

#plot for class
# sns.barplot(x=survival_by_class.index, y=survival_by_class.values)
# plt.xlabel("Passenger Class")
# plt.ylabel("Survival Rate")
# plt.title("Survival Rate by Class")

#plot for gender
# sns.barplot(x=survival_by_gender.index, y=survival_by_gender.values)
# plt.xlabel("Gender")
# plt.ylabel("Survival Rate")
# plt.title("Survival Rate by Gender")

#plot for AgeGroup
# sns.barplot(x=survival_by_agegroup.index, y=survival_by_agegroup.values)
# plt.xlabel("Age Group")
# plt.ylabel("Survival Rate")
# plt.title("Survival Rate by AgeGroup")
#
# for i, rate in enumerate(survival_by_agegroup.values):
#     plt.text(i, rate+2, f"{rate:.1f}%", ha='center', fontsize=12)

#percentage above bar
# for i, rate in enumerate(survival_by_class.values):
#     plt.text(i, rate + 2, f"{rate:.1f}%", ha='center', fontsize=12)
# for i, rate in enumerate(survival_by_gender.values):
#     plt.text(i, rate + 2, f"{rate:.1f}%", ha='center', fontsize=12)

#     plt.ylim(0, 100)
# plt.show()

# class_survived = df.groupby(['Pclass'])['Survived'].value_counts(normalize=True)*100
# class_survived = class_survived.rename('Percentage').reset_index()
#
# plt.figure(figsize=(8,6))
# sns.barplot(x='Pclass', y='Percentage', hue='Survived', data=class_survived)
# plt.title('Survival Rate by Passenger Class')
# plt.xlabel('Passenger Class')
# plt.ylabel('Survival Rate %')
# plt.show()

# gender_survived = df.groupby(['Sex'])['Survived'].value_counts(normalize=True)*100
# gender_survived = gender_survived.rename('Percentage').reset_index()
#
# plt.figure(figsize=(8,6))
# sns.barplot(x='Sex', y='Percentage', hue='Survived', data=gender_survived)
# plt.title('Survival Rate by Gender')
# plt.xlabel('Gender')
# plt.ylabel('Survival rate %')
# plt.show()

# familysize_survived = df.groupby(['Family Size'])['Survived'].value_counts(normalize=True)*100
# familysize_survived = familysize_survived.rename('Percentage').reset_index()
#
# plt.figure(figsize=(8,6))
# sns.barplot(x='Family Size', y='Percentage', hue='Survived', data=familysize_survived)
# plt.title('Survival rate by FamilySize')
# plt.xlabel('Family Size')
# plt.ylabel('Survival Rate %')
# plt.show()








# pd.set_option('display.max_columns', None)
# df = pd.read_csv("Titanic-Dataset2.csv")
#print(df.head())
# df.info()

#1st check of missing values
# print(df.isnull().sum())
#Age = 177MV, Cabin = 687MV

#fill the missing age with median age
# first find the median
# median_age = df ['Age'].median()
# print("Median Age:", median_age)

#then fill the missing values in Age with median (28)
#df['Age'].fillna(df['Age'].median(), inplace=True)
# df['Age'] = df['Age'].fillna(28)

#2nd Check of missing values
# print(df.isnull().sum())

#Cabin has 687 missing values, drop this column as it is not relative to the analysis
# df = df.drop(['Cabin'], axis=1)

#Ticket column with different format of number
#checking if its useable

# df['Ticket'].nunique()
# print(df['Ticket'].nunique())
#681 unique numbers, dropping this as well, for wealth analysis use passengerclass instead of ticket
# df = df.drop(['Ticket'], axis=1)

#Embarked also has some missing values, dropping the rows
# df.dropna(subset=['Embarked'], inplace=True)

#formatting categorical variables
# Sex: Male = 1, Female = 0
# df['Sex'] = df['Sex'].map({'male':1, 'female':0})

#Name has unique titles and names, they are not relevant to the analysis.
# If title represent wealth then use fare. dropping the column
# df = df.drop(['Name'], axis=1)

#data processing
#Adding FamilySize column in excel
#adding AgeGroup column
#age range 0-2: Infant, 3-12: Child, 13-17: Teen, 18-30: Young Adult, 31-50: Adult, 51+: Elderly
# bins = [0, 2, 12, 17, 30, 50, 100]
# labels = ['Infant', 'Child', 'Teen', 'Young Adult', 'Adult', 'Elderly']
#
# df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
# df['AgeGroup'].value_counts()
# print(df.head())
#
# df['Title'] = df['Name'].str.extract(r",\s*([^\.]+)



