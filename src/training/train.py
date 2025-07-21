import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
data = pd.read_csv('data/raw/remote_work_health.csv')

# Preprocessing
data_model = data.copy()
# Impute Mental_Health_Status with mode
mental_imputer = SimpleImputer(strategy='most_frequent')
data_model['Mental_Health_Status'] = mental_imputer.fit_transform(data_model[['Mental_Health_Status']]).ravel()
# Impute Physical_Health_Issues with 'None'
data_model['Physical_Health_Issues'].fillna('None', inplace=True)
# Encode categorical variables
categorical_cols = ['Gender', 'Region', 'Industry', 'Job_Role', 'Work_Arrangement', 
                    'Mental_Health_Status', 'Physical_Health_Issues', 'Salary_Range', 'Burnout_Level']
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data_model[col] = label_encoders[col].fit_transform(data_model[col])
# Scale numerical features
numerical_cols = ['Age', 'Hours_Per_Week', 'Work_Life_Balance_Score', 'Social_Isolation_Score']
scaler = StandardScaler()
data_model[numerical_cols] = scaler.fit_transform(data_model[numerical_cols])

# Feature Engineering
data_model['Work_Load'] = data_model['Hours_Per_Week'] * data_model['Social_Isolation_Score']
data_model['Health_Stress_Index'] = data_model['Work_Life_Balance_Score'] * data_model['Mental_Health_Status']
# Bin Age into Age_Group
data_model['Age_Group'] = pd.cut(data['Age'], bins=[20, 30, 40, 50, 60, 70], labels=[0, 1, 2, 3, 4], include_lowest=True)
data_model['Age_Group'] = data_model['Age_Group'].cat.codes

# Prepare feature matrix and target
X = data_model.drop(['Burnout_Level', 'Survey_Date'], axis=1)
y = data_model['Burnout_Level']

# Train Random Forest Classifier
rf_model = RandomForestClassifier(
    class_weight={0: 1.0, 1: 1.5, 2: 0.8},
    max_depth=20,
    min_samples_split=5,
    n_estimators=100,
    random_state=42
)
rf_model.fit(X, y)

# Save model and preprocessing objects
os.makedirs('models', exist_ok=True)
joblib.dump(rf_model, 'models/model.joblib')
joblib.dump(label_encoders, 'models/label_encoders.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(mental_imputer, 'models/mental_imputer.joblib')
