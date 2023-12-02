# App-Rating-Prediction-for-Google-Play-Store
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer

# Load the data file using pandas
data = pd.read_csv("C:/MURTHY/DATA_ANALYTICS/Assessment/Python/1569582940_googleplaystore/googleplaystore.csv")

# Check for null values in the data and get the number of null values for each column
null_counts = data.isnull().sum()

# Drop records with nulls in any of the columns
data = data.dropna()


# Convert 'Size' column to numeric
data['Size'] = data['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
data['Size'] = data['Size'].apply(lambda x: str(x).replace('k', '') if 'k' in str(x) else x)
data['Size'] = pd.to_numeric(data['Size'], errors='coerce')
data['Size'] = data['Size'].apply(lambda x: x * 1000 if x > 1 else x)  # Convert MB to KB

# Convert 'Reviews' column to numeric
data['Reviews'] = data['Reviews'].astype(int)

# Convert 'Installs' field to numeric
data['Installs'] = data['Installs'].apply(lambda x: int(x.replace('+', '').replace(',', '')))

# Convert 'Price' field to numeric
data['Price'] = data['Price'].apply(lambda x: float(x.replace('$', '')))

# Now you have a cleaned and preprocessed dataset
# Drop rows with rating outside the range [1, 5]
data = data[(data['Rating'] >= 1) & (data['Rating'] <= 5)]

# Drop rows where reviews are more than installs
data = data[data['Reviews'] <= data['Installs']]

# Drop rows where the app is free but the price is greater than 0
data = data[~((data['Type'] == 'Free') & (data['Price'] > 0))]

# Now you have a dataset with rows that pass the sanity checks

# Set the style of seaborn for better visualization
sns.set(style="whitegrid")

# Boxplot for Price
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['Price'])
plt.title('Boxplot for Price')
plt.xlabel('Price')
plt.show()

# Boxplot for Reviews
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['Reviews'])
plt.title('Boxplot for Reviews')
plt.xlabel('Reviews')
plt.show()

# Histogram for Rating
plt.figure(figsize=(10, 6))
plt.hist(data['Rating'], bins=20, edgecolor='black')
plt.title('Histogram for Rating')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Histogram for Size
plt.figure(figsize=(10, 6))
plt.hist(data['Size'], bins=20, edgecolor='black')
plt.title('Histogram for Size')
plt.xlabel('Size')
plt.ylabel('Frequency')
plt.show()

# Check records with very high price
high_price_apps = data[data['Price'] > 200]
print(high_price_apps)

# Drop records with price > 200
data = data[data['Price'] <= 200]

# Drop records with more than 2 million reviews
data = data[data['Reviews'] <= 2000000]

# Drop records with very high number of installs
install_percentiles = data['Installs'].quantile([0.10, 0.25, 0.50, 0.70, 0.90, 0.95, 0.99])
install_cutoff = install_percentiles[0.99]
data = data[data['Installs'] <= install_cutoff]

# Now you have the dataset with outliers removed

# Scatter plot for Rating vs. Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Price', y='Rating', data=data)
plt.title('Rating vs. Price')
plt.xlabel('Price')
plt.ylabel('Rating')
plt.show()

# Scatter plot for Rating vs. Size
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Size', y='Rating', data=data)
plt.title('Rating vs. Size')
plt.xlabel('Size')
plt.ylabel('Rating')
plt.show()

# Scatter plot for Rating vs. Reviews
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Reviews', y='Rating', data=data)
plt.title('Rating vs. Reviews')
plt.xlabel('Reviews')
plt.ylabel('Rating')
plt.show()

# Boxplot for Rating vs. Content Rating
plt.figure(figsize=(10, 6))
sns.boxplot(x='Content Rating', y='Rating', data=data)
plt.title('Rating vs. Content Rating')
plt.xlabel('Content Rating')
plt.ylabel('Rating')
plt.xticks(rotation=45)
plt.show()

# Boxplot for Rating vs. Category
plt.figure(figsize=(15, 8))
sns.boxplot(x='Category', y='Rating', data=data)
plt.title('Rating vs. Category')
plt.xlabel('Category')
plt.ylabel('Rating')
plt.xticks(rotation=90)
plt.show()


# Create a copy of the dataframe
inp1 = data.copy()

# Apply log transformation to Reviews and Installs
inp1['Reviews'] = np.log1p(inp1['Reviews'])
inp1['Installs'] = np.log1p(inp1['Installs'])

# Drop unnecessary columns
columns_to_drop = ['App', 'Last Updated', 'Current Ver', 'Android Ver']
inp1.drop(columns=columns_to_drop, inplace=True)

# Create a new dataframe (inp2) with dummy columns
# Drop the 'Type' column before performing dummy encoding
inp1.drop(columns=['Type'], inplace=True)
inp2 = pd.get_dummies(inp1, columns=['Category', 'Genres', 'Content Rating'], drop_first=True)




# Splitting the data
X = inp2.drop(columns=['Rating'])
y = inp2['Rating']

# Splitting into train and test sets (70-30 split)
df_train, df_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Separate df_train into X_train and y_train
X_train = df_train
y_train = y_train

# Separate df_test into X_test and y_test
X_test = df_test
y_test = y_test

# Initialize the SimpleImputer
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on X_train to impute missing values
X_train_imputed = imputer.fit_transform(X_train)

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model on the imputed train set
model.fit(X_train_imputed, y_train)

# Predict the ratings on the train set
y_train_pred = model.predict(X_train_imputed)

# Calculate R-squared on the train set
r2_train = r2_score(y_train, y_train_pred)

# Print the R-squared on the train set
print("R-squared on the train set:", r2_train)

# Transform the imputer on X_test to impute missing values
X_test_imputed = imputer.transform(X_test)

# Predict the ratings on the test set
y_test_pred = model.predict(X_test_imputed)

# Calculate R-squared on the test set
r2_test = r2_score(y_test, y_test_pred)

# Print R-squared on the test set
print("R-squared on the test set:", r2_test)
 
