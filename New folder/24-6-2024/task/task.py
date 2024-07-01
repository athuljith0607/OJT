import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Load the dataset from a CSV file into a Pandas DataFrame
data = pd.read_csv('sample_dataset.csv')

# Clean the column names by removing spaces and commas
data.columns = [col.strip().replace(' ', '_').replace(',', '') for col in data.columns]

print("First few rows of the dataset:")
print(data.head())

# 2. Generate summary statistics
summary_stats = data.describe()
mean_sepal_length = summary_stats.loc['mean', 'Sepal_Length_(cm)']
std_sepal_length = summary_stats.loc['std', 'Sepal_Length_(cm)']
print("\nSummary Statistics:")
print(summary_stats)
print(f"\nMean Sepal Length: {mean_sepal_length}")
print(f"Standard Deviation Sepal Length: {std_sepal_length}")

# 3. Check for any missing values
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Handling missing values (if any)
# For the purpose of this example, we'll assume there are no missing values.
# If there were missing values, we could handle them by imputation or removing rows/columns.

# 4. Convert the species labels to numerical values
species_mapping = {'FlowerA': 0, 'FlowerB': 1, 'FlowerC': 2}
data['Species'] = data['Species'].map(species_mapping)
print("\nDataset after mapping species to numerical values:")
print(data.head())

# 5. Split the dataset into training and testing sets
X = data.drop('Species', axis=1)
y = data['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 6. Train a decision tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Visualize the trained decision tree
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['FlowerA', 'FlowerB', 'FlowerC'])
plt.title("Decision Tree")
plt.show()

# 8. Predict the species for the testing data and compute the accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# 9. Generate a classification report and a confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
