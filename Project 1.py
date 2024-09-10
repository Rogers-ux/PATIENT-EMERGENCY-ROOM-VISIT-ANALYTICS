#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load dataset
movies=pd.read_csv('movies.csv')

#missing data
# print(movies.isnull().sum())

#Handle the numerical columns
#1st method
# movies['score'].fillna(movies['score'].mean(),inplace=True)
# movies['votes'].fillna(movies['votes'].mean(),inplace=True)
# movies['budget'].fillna(movies['budget'].mean(),inplace=True)
# movies['gross'].fillna(movies['gross'].mean(),inplace=True)
# movies['runtime'].fillna(movies['runtime'].mean(),inplace=True)
#
# #2nd method for loop
numerical_columns=['score','votes','budget','gross','runtime']

for column in numerical_columns:
    movies[column].fillna(movies[column].mean(),inplace=True)


#Handle the categorical data
categorical_columns=['rating','released','writer','star','country','company']

for column in categorical_columns:
    movies[column].fillna('unknown',inplace=True)

#check data types
movies['budget']=movies['budget'].astype('int64')
movies['gross']=movies['gross'].astype('int64')

#create correct year column
movies["correct_year"]=movies["released"].astype("str").str[:4]

#sort  by gross revenue
movies.sort_values(by='gross',inplace=False,ascending=False)

#drop duplicates
movies.drop_duplicates()

#finding correlations

#visualize
# plt.figure(figsize=(12,6))
# plt.scatter(movies['budget'],movies['gross'])
# plt.xlabel('Gross Revenue')
# plt.ylabel('Budget')
# plt.title("How gross Revenue varies by budget")
# plt.show()

#Regression plot

# sns.regplot(x=movies['budget'],y=movies['gross'],data=movies)
# plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Set the style for the plot
sns.set(style="whitegrid")

# Create a larger figure
plt.figure(figsize=(12, 6))

# Plot the regression plot with enhancements
sns.regplot(x='budget', y='gross', data=movies, scatter_kws={'alpha':0.6}, line_kws={"color":"red", "lw":2})

# Adding a title and labels
plt.title('Relationship between Movie Budget and Gross Earnings', fontsize=16)
plt.xlabel('Budget (in dollars)', fontsize=14)
plt.ylabel('Gross Earnings (in dollars)', fontsize=14)

# Rotate x-axis labels if the numbers are large and overlapping
plt.xticks(rotation=45)

# Show the plot
# plt.show()

#----------CORRELATIONS
# Select only numerical columns from the DataFrame
numerical_columns = movies.select_dtypes(include=[np.number])

#pearson
pearson_corr=numerical_columns.corr(method='pearson')

print("Pearson Correlation Matrix:")
print(pearson_corr)

# Spearman correlation
spearman_corr = numerical_columns.corr(method='spearman')
print("\nSpearman Correlation Matrix:")
print(spearman_corr)

# Kendall correlation
kendall_corr = numerical_columns.corr(method='kendall')
print("\nKendall Correlation Matrix:")
print(kendall_corr)

#Pearson visual
plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title('Pearson Correlation Matrix')
plt.show()

#spearman correlation
plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title('Spearman Correlation Matrix')
plt.show()

#kendall correlation
plt.figure(figsize=(10, 8))
sns.heatmap(kendall_corr, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title('Kendall Correlation Matrix')
plt.show()

