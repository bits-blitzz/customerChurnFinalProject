import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import pickle

print("Script started...")

# ---Load Data ---
# We load the CSV file into a pandas DataFrame.
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: The data file was not found. Make sure 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is in the same folder as the script.")
    exit()

# ---Clean and Prepare Data ---
# Convert 'TotalCharges' to numbers, handling errors.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Remove rows with missing values (which are few).
df.dropna(subset=['TotalCharges'], inplace=True)

# Drop the customerID column as it's not a feature.
df_processed = df.drop('customerID', axis=1)

# Convert 'Yes'/'No' answers into 1s and 0s for the model to understand.
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df_processed[col] = df_processed[col].apply(lambda x: 1 if x == 'Yes' else 0)

# Identify which columns are numerical and which are categorical (text-based).
categorical_features = df_processed.select_dtypes(include=['object']).columns
numerical_features = df_processed.select_dtypes(include=['int64', 'float64']).drop(['Churn'], axis=1).columns

# ---Define Preprocessing Steps ---
# We create a "preprocessor" that will handle all data preparation automatically.
# For numerical features, we scale them (StandardScaler).
# For categorical features, we convert them into numbers (OneHotEncoder).
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# ---Define the Model ---
# We are using an XGBoost Classifier, which is a powerful and popular model.
model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# ---Create the Full Pipeline ---
# A pipeline chains the preprocessing and the model together.
# When we give it data, it will automatically preprocess it and then feed it to the model.
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])

# --- Train the Model ---
# We separate our data into features (X) and the target we want to predict (y, which is 'Churn').
X = df_processed.drop('Churn', axis=1)
y = df_processed['Churn']

# We train the pipeline on our data. The .fit() command is where the learning happens.
print("Training the model... This might take a moment.")
pipeline.fit(X, y)
print("Model training complete.")

# --- Save the Model ---
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved successfully as 'churn_model.pkl'!")