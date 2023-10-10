import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense

# Step 1: Load the Excel file
excel_file_path = 'data.trajeto.xlsx'
sheet_name = 'data.trajeto'

# Load data from Excel
try:
    excel_data = pd.read_excel(excel_file_path)#, sheet_name=sheet_name)
except FileNotFoundError:
    print(f"File {excel_file_path} not found.")
    exit(1)
except Exception as e:
    print(f"Error loading data from Excel: {str(e)}")
    exit(1)

# Step 2: Check if the required columns are present
required_columns = ['Velocidade_km_h', 'cluster']

missing_columns = [col for col in required_columns if col not in excel_data.columns]
if missing_columns:
    print(f"Required columns are missing: {missing_columns}")
    exit(1)

# Step 3: Prepare the data for neural network
X = excel_data[required_columns[:-1]]  # Features
y = excel_data['cluster']       # Target variable

# Encode categorical target variable (output_cluster) to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.02, random_state=42)

print(type(y_train[0]), type(y_test))
# Step 5: Standardize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Build the neural network model
model = Sequential()
# Input layer
model.add(Dense(16, input_dim=20, activation=’relu’))
model.add(Dense(12, activation=’relu’))
model.add(Dense(4, activation=’softmax’))
# Output layer
#model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer = "adam", metrics=['accuracy'])

# Step 7: Train the neural network
model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=3)
print(model.summary())

# Step 8: Evaluate the model on the test data
accuracy = model.evaluate(X_test, y_test)[1]
print(X_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
model.save('mymodel.h5')
# Now you have a trained neural network that can classify clusters based on the specified columns in the Excel file.
