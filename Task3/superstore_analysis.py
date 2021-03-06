import pandas as pd

# Importing Data
superstore = pd.read_csv('SampleSuperstore.csv')

# Looking at first 5 rows
print(superstore.head()) # 5 * 13 size

# Looking at columns
print(superstore.columns)
# Index(['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Postal Code',
#        'Region', 'Category', 'Sub-Category', 'Sales', 'Quantity', 'Discount',
#        'Profit'],
#       dtype='object')

print(superstore.shape) # (9994, 13)

print(superstore.info())

# Data columns (total 13 columns):
#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   Ship Mode     9994 non-null   object
#  1   Segment       9994 non-null   object
#  2   Country       9994 non-null   object
#  3   City          9994 non-null   object
#  4   State         9994 non-null   object
#  5   Postal Code   9994 non-null   int64
#  6   Region        9994 non-null   object
#  7   Category      9994 non-null   object
#  8   Sub-Category  9994 non-null   object
#  9   Sales         9994 non-null   float64
#  10  Quantity      9994 non-null   int64
#  11  Discount      9994 non-null   float64
#  12  Profit        9994 non-null   float64

print(superstore.describe())

#  Postal Code         Sales     Quantity     Discount       Profit
# count   9994.000000   9994.000000  9994.000000  9994.000000  9994.000000
# mean   55190.379428    229.858001     3.789574     0.156203    28.656896
# std    32063.693350    623.245101     2.225110     0.206452   234.260108
# min     1040.000000      0.444000     1.000000     0.000000 -6599.978000
# 25%    23223.000000     17.280000     2.000000     0.000000     1.728750
# 50%    56430.500000     54.490000     3.000000     0.200000     8.666500
# 75%    90008.000000    209.940000     5.000000     0.200000    29.364000
# max    99301.000000  22638.480000    14.000000     0.800000  8399.976000

# Finding unique Quantity
print(superstore['Quantity'].unique())

# Finding value_counts() of unique Quantity
print(superstore['Quantity'].value_counts())
# 3     2409
# 2     2402
# 5     1230
# 4     1191
# 1      899
# 7      606
# 6      572
# 9      258
# 8      257
# 10      57
# 11      34
# 14      29
# 13      27
# 12      23
# Name: Quantity, dtype: int64

# Suppose i want to see cities and then unique cities
print(superstore['City'][:10]) # print cities
# 0          Henderson
# 1          Henderson
# 2        Los Angeles
# 3    Fort Lauderdale
# 4    Fort Lauderdale
# 5        Los Angeles
# 6        Los Angeles
# 7        Los Angeles
# 8        Los Angeles
# 9        Los Angeles
# Name: City, dtype: object
print(len(superstore['City'].value_counts())) # returns 531
print(superstore['City'].value_counts()[:50])
# DATA VISUALISATION
import seaborn as sns
import matplotlib.pyplot as plt

# Plotting heatmap to show whether there is any null value or not
print(sns.heatmap(superstore.isnull(), cbar=True, yticklabels=False, cmap='viridis'))
plt.title('Superstore Null/Non-null Value Representation')
plt.show()

# Plotting heatmap for correlation with annot=False
sns.heatmap(superstore.corr(), annot=False, cmap='Greens')
plt.title('Superstore Correlation(annot=False)')
plt.show()
# Positive correlation is represented by dark shades and negative correlation
# by lighter shades.

# Plotting heatmap for correlation with annot=True
sns.heatmap(superstore.corr(), annot=True, cmap='Greens')
plt.title('Superstore Correlation(annot=True)')
plt.show()

# plotting Categories against profit
print(superstore['Category'].value_counts())
# Office Supplies    6026
# Furniture          2121
# Technology         1847
superstore.plot(kind='scatter', x='Category', y='Profit')
plt.show() # Shows we need to invest more in technology as it's more beneficial
superstore.plot(kind='scatter', x='Profit', y='Sub-Category')
plt.show() # Binders, machines has high profit


# Plotting Region against Profit
superstore.plot(kind='scatter', x='Region', y='Profit')
plt.show() # Ignoring outliers, it's min in west, more in central and east region

# Plotting Sales against profit
superstore.plot(kind='scatter', x='Sales', y='Profit')
plt.show() # Removing outlier, profit increases with sales

# Plotting Quantity against sales
superstore.plot(kind='scatter', x='Quantity', y='Sales')
plt.show() # Quantity between 2-6 has max sales

# Plot of State against profit
superstore.plot(kind='scatter', x='Profit', y='State')
plt.show() # Profit is higher in major cities

superstore.plot(kind='scatter', x='Discount', y='Profit')
plt.show() # Profit is high when discount is minimum

