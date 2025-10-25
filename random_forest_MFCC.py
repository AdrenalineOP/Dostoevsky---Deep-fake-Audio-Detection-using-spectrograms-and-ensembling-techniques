import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

base_model = RandomForestClassifier(random_state=42)
base_model.fit(X_train_scaled, y_train)
base_pred = base_model.predict(X_test_scaled)
base_accuracy = accuracy_score(y_test, base_pred)

param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

best_grid_model = grid_search.best_estimator_
grid_pred = best_grid_model.predict(X_test_scaled)
grid_accuracy = accuracy_score(y_test, grid_pred)

param_dist = {
    'n_estimators': np.arange(50, 500, 50),
    'max_depth': [None] + list(np.arange(5, 50, 5)),
    'min_samples_split': np.arange(2, 20, 2),
    'min_samples_leaf': np.arange(1, 10, 1),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
    'max_samples': [0.6, 0.7, 0.8, 0.9, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    random_state=42,
    verbose=1
)

random_search.fit(X_train_scaled, y_train)

best_random_model = random_search.best_estimator_
random_pred = best_random_model.predict(X_test_scaled)
random_accuracy = accuracy_score(y_test, random_pred)

cv_scores_base = cross_val_score(base_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_scores_grid = cross_val_score(best_grid_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_scores_random = cross_val_score(best_random_model, X_train_scaled, y_train, cv=5, scoring='accuracy')

print(f"Base Model Accuracy: {base_accuracy:.4f}")
print(f"Grid Search Best Accuracy: {grid_accuracy:.4f}")
print(f"Random Search Best Accuracy: {random_accuracy:.4f}")
print(f"\nBase Model CV Mean Accuracy: {cv_scores_base.mean():.4f} (+/- {cv_scores_base.std() * 2:.4f})")
print(f"Grid Search CV Mean Accuracy: {cv_scores_grid.mean():.4f} (+/- {cv_scores_grid.std() * 2:.4f})")
print(f"Random Search CV Mean Accuracy: {cv_scores_random.mean():.4f} (+/- {cv_scores_random.std() * 2:.4f})")
print(f"\nGrid Search Best Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nRandom Search Best Parameters:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest Model Classification Report:")
best_model = best_random_model if random_accuracy >= grid_accuracy else best_grid_model
final_pred = best_model.predict(X_test_scaled)
print(classification_report(y_test, final_pred))

feature_importance = pd.DataFrame({
    'feature': range(X.shape[1]),
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
models = ['Base', 'Grid Search', 'Random Search']
accuracies = [base_accuracy, grid_accuracy, random_accuracy]
plt.bar(models, accuracies, color=['blue', 'orange', 'green'])
plt.title('Model Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

plt.subplot(1, 3, 2)
cv_means = [cv_scores_base.mean(), cv_scores_grid.mean(), cv_scores_random.mean()]
cv_stds = [cv_scores_base.std(), cv_scores_grid.std(), cv_scores_random.std()]
plt.bar(models, cv_means, yerr=cv_stds, capsize=5, color=['blue', 'orange', 'green'])
plt.title('Cross-Validation Performance')
plt.ylabel('Mean CV Accuracy')
plt.ylim(0, 1)

plt.subplot(1, 3, 3)
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), [f'Feature {i}' for i in top_features['feature']])
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')

plt.tight_layout()
plt.show()

conf_matrix = confusion_matrix(y_test, final_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Best Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

import pickle
import joblib

with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

joblib.dump(best_model, 'random_forest_model.joblib')

joblib.dump(scaler, 'scaler.joblib')

with open('random_forest_model.pkl', 'rb') as file:
    loaded_model_pkl = pickle.load(file)

loaded_model_joblib = joblib.load('random_forest_model.joblib')

loaded_scaler = joblib.load('scaler.joblib')

X_test_scaled_loaded = loaded_scaler.transform(X_test)
predictions_loaded = loaded_model_joblib.predict(X_test_scaled_loaded)
loaded_accuracy = accuracy_score(y_test, predictions_loaded)

print(f"Original model accuracy: {random_accuracy:.4f}")
print(f"Loaded model accuracy: {loaded_accuracy:.4f}")

model_metadata = {
    'model_type': 'RandomForestClassifier',
    'best_params': random_search.best_params_,
    'accuracy': random_accuracy,
    'feature_count': X.shape[1],
    'classes': best_model.classes_.tolist()
}

with open('model_metadata.pkl', 'wb') as file:
    pickle.dump(model_metadata, file)

print("Model saved successfully!")
print(f"Best parameters: {random_search.best_params_}")
print(f"Model accuracy: {random_accuracy:.4f}")