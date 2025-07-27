import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle

print("Model builder starting...")

# Load dataset
df = pd.read_csv("Dataset.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace('\xa0', ' ')

# Print column names for debug
print("\nAvailable columns in dataset:")
print(df.columns.tolist())

# Drop rows with missing values
df.dropna(inplace=True)

# Updated feature names as per your request
selected_features = ['Votes', 'Restaurant ID', 'Is delivering now',
                     'Price range', 'Has Online delivery', 'Has Table booking']

required_columns = selected_features + ['Aggregate rating']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise KeyError(f"Missing columns in dataset: {missing_columns}")

# Subset the DataFrame
df = df[required_columns]

# Encode categorical features
label_cols = ['Is delivering now', 'Has Online delivery', 'Has Table booking']
encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Define features and target
X = df.drop('Aggregate rating', axis=1)
y = df['Aggregate rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and encoders
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("✅ Model training complete and saved as model.pkl and encoders.pkl")

