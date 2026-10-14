import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

print("🔄 Initializing Dynamic SensorMind AI Training Pipeline...")

# 1. Load Dataset
data_path = os.path.join('data', 'predictive_maintenance.csv')
df = pd.read_csv(data_path)

# Let's inspect the exact spelling of your file columns
print("\n📋 Your file columns are exactly:", list(df.columns))

# Clean bracket labels out to avoid matching bugs
df.columns = df.columns.str.replace(r'\[.*\]', '', regex=True).str.strip()

# Dynamic Target Column Mapper (Finds 'Machine failure' or 'Target' case insensitively)
target_column = None
for col in df.columns:
    if col.lower() in ['machine failure', 'target', 'failure']:
        target_column = col
        break

if target_column is None:
    raise KeyError("Could not automatically locate your target failure column! Check column printout above.")

# 2. Advanced Feature Engineering 
df['thermal_dissipation_delta'] = df['Process temperature'] - df['Air temperature']
df['mechanical_power_index'] = df['Rotational speed'] * df['Torque']
df['wear_strain_factor'] = df['Tool wear'] * df['Torque']

# 3. Separate Input Matrices from target label
features = [
    'Air temperature', 'Process temperature', 'Rotational speed', 
    'Torque', 'Tool wear', 'thermal_dissipation_delta', 
    'mechanical_power_index', 'wear_strain_factor'
]

X = df[features]
y = df[target_column]

print(f"🎯 Target label column locked: '{target_column}'")

# 4. Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Execute Tuned Ensemble Training
model = RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate Model Metrics
predictions = model.predict(X_test)
print(f"\n✅ Training Complete. Validation Engine Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\n--- Model Performance Metrics ---")
print(classification_report(y_test, predictions, target_names=['Operational', 'Failed']))

# 7. Serialize Intelligence Asset
joblib.dump(model, 'maintenance_model.pkl')
print("📦 'maintenance_model.pkl' successfully compiled to workspace root folder.")
