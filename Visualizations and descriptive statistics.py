import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load data
whiteData = pd.read_csv('winequality-white.csv', delimiter=";")
redData = pd.read_csv('winequality-red.csv', delimiter=";")

########## Some basic desciptive statistics ##########
redData['quality'].mean()
whiteData['quality'].mean()

# Gonna do some more stuff in here


########## Correlations ##########
# Get Correlations
white_corr = whiteData.corr()
red_corr = redData.corr()


# Correlations with quality, greatest to least
# Extract correlations of 'quality' with other variables
white_quality_correlations = white_corr['quality'].sort_values(ascending=False)
red_quality_correlations = red_corr['quality'].sort_values(ascending=False)

# Remove the correlation of 'quality' with itself (which will be 1.0)
white_quality_correlations = white_quality_correlations.drop('quality')
red_quality_correlations = red_quality_correlations.drop('quality')

# Print correlations
print("Quality correlations for white wine:\n")
print(white_quality_correlations, "\n\n")
print("Quality correlations for red wine:\n")
print(red_quality_correlations)


# Create heatmaps of correlations
sns.set(style="white")

plt.figure(figsize=(8, 6))
sns.heatmap(white_corr, annot=True, cmap='RdYlBu', fmt=".2f", linewidths=0.5)
plt.title('White Wine Correlation Matrix')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(red_corr, annot=True, cmap='plasma', fmt=".2f", linewidths=0.5)
plt.title('Red Wine Correlation Matrix')
plt.show()


########## Graphs ##########

### White wine quality histogram ###
white_quality_values = whiteData['quality']

# Create the histogram
plt.hist(white_quality_values, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], color='lightyellow', edgecolor='black')

# Adding labels and title
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.title('Quality distribution of white wines')

# Display the plot
plt.show()


### Red wine quality histogram ###
red_quality_values = redData['quality']

# Create the histogram
plt.hist(red_quality_values, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], color='red', edgecolor='black')

# Adding labels and title
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.title('Quality distribution of red wines')

# Display the plot
plt.show()


### Scatter plot of white quality and alcohol ###
plt.figure(figsize=(8, 6))
sns.scatterplot(x='alcohol', y='quality', data=whiteData)
sns.regplot(x='alcohol', y='quality', data=whiteData, scatter=False, color='red')  # Add trendline
plt.title('Scatter Plot of Alcohol vs Quality with Trendline')
plt.grid(True)
plt.show()

### Scatter plot of white quality and density ###
plt.figure(figsize=(8, 6))
sns.scatterplot(x='density', y='quality', data=whiteData)
sns.regplot(x='alcohol', y='quality', data=whiteData, scatter=False, color='red')  # Add trendline
plt.title('Scatter Plot of Density vs Quality with Trendline')
plt.grid(True)
plt.show()

### Scatter plot of red quality and alcohol ###
plt.figure(figsize=(8, 6))
sns.scatterplot(x='alcohol', y='quality', data=redData)
sns.regplot(x='alcohol', y='quality', data=redData, scatter=False, color='red')  # Add trendline
plt.title('Scatter Plot of Alcohol vs Quality with Trendline')
plt.grid(True)
plt.show()

### Scatter plot of red quality and volatile acidity ###
plt.figure(figsize=(8, 6))
sns.scatterplot(x='volatile acidity', y='density', data=redData)
sns.regplot(x='volatile acidity', y='quality', data=redData, scatter=False, color='red')  # Add trendline
plt.title('Scatter Plot of Volatile Acidity vs Quality with Trendline')
plt.grid(True)
plt.show()





