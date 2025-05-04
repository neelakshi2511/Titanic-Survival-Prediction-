import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# basic information about the dataset
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn information:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Exploratory Data Analysis 
plt.figure(figsize=(12, 5))

# Plot 1: Survival count
plt.subplot(1, 3, 1)
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')

# Plot 2: Survival by gender
plt.subplot(1, 3, 2)
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival by Gender')

# Plot 3: Survival by passenger class
plt.subplot(1, 3, 3)
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival by Class')

plt.tight_layout()
plt.show()

# Plot 1: Age distribution
plt.subplot(2, 2, 1)
sns.histplot(df['Age'].dropna(), kde=True)
plt.title('Age Distribution')

# Plot 2: Fare distribution
plt.subplot(2, 2, 2)
sns.histplot(df['Fare'].dropna(), kde=True)
plt.title('Fare Distribution')

# Plot 3: Age vs. Fare with survival coloring
plt.subplot(2, 2, 3)
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title('Age vs. Fare (by Survival)')

# Plot 4: Correlation heatmap
plt.subplot(2, 2, 4)
corr = df.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')

plt.tight_layout()
plt.show()

# Data Preprocessing
# Select features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Define numeric and categorical features
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

# Define preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create modeling pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature importance
if hasattr(model[-1], 'feature_importances_'):
    # Get feature names
    ohe_feature_names = []
    for name, transformer, features in preprocessor.transformers_:
        if name == 'cat':
            # Get the one-hot encoded feature names
            ohe = transformer.named_steps['onehot']
            ohe_feature_names.extend([f"{feature}_{category}" for feature, categories in 
                                     zip(features, ohe.categories_) for category in categories])
        else:
            ohe_feature_names.extend(features)
    
    # Plot feature importance
    importances = model[-1].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [ohe_feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

# Hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("\nHyperparameter Tuning Results:")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Final model with best parameters
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("\nFinal Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

# Create a function to predict survival
def predict_survival(passenger_info):
    """
    Predict survival based on passenger information
    
    Parameters:
    passenger_info (dict): Dictionary containing passenger information
    
    Returns:
    int: 0 for not survived, 1 for survived
    float: Probability of survival
    """
    # Convert to DataFrame
    passenger_df = pd.DataFrame([passenger_info])
    
    # Make prediction
    prediction = best_model.predict(passenger_df)[0]
    probability = best_model.predict_proba(passenger_df)[0][1]
    
    return prediction, probability

# Example usage
sample_passenger = {
    'Pclass': 1,
    'Sex': 'female',
    'Age': 30,
    'SibSp': 0,
    'Parch': 0,
    'Fare': 100,
    'Embarked': 'C'
}

prediction, probability = predict_survival(sample_passenger)
print("\nSample Prediction:")
print(f"Passenger would {'survive' if prediction == 1 else 'not survive'}")
print(f"Survival probability: {probability:.4f}")

# Conclusion
print("\nConclusion:")
print("We have successfully built a machine learning model to predict Titanic passenger survival.")
print(f"The model achieves an accuracy of {accuracy_score(y_test, y_pred_best):.4f} on the test set.")
print("The most important features for prediction are:")
for i in np.argsort(best_model.named_steps['classifier'].feature_importances_)[::-1][:5]:
    print(f"- {ohe_feature_names[i]}")
