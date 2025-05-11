import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import argparse

# Load the dataset
df = pd.read_csv("penguins_lter_clean.csv")

# Select features and target
features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
target = 'Species'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model Evaluation:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "penguin_species_model.pkl")
print("Model saved as 'penguin_species_model.pkl'.")

# Command-line interface
def predict_species(culmen_length, culmen_depth, flipper_length, body_mass):
    model = joblib.load("penguin_species_model.pkl")
    input_data = pd.DataFrame([[culmen_length, culmen_depth, flipper_length, body_mass]], columns=features)
    prediction = model.predict(input_data)
    return prediction[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Penguin Species based on physical traits.")
    parser.add_argument("--culmen_length", type=float, required=True, help="Culmen Length (mm)")
    parser.add_argument("--culmen_depth", type=float, required=True, help="Culmen Depth (mm)")
    parser.add_argument("--flipper_length", type=float, required=True, help="Flipper Length (mm)")
    parser.add_argument("--body_mass", type=float, required=True, help="Body Mass (g)")

    args = parser.parse_args()
    species = predict_species(args.culmen_length, args.culmen_depth, args.flipper_length, args.body_mass)
    print(f"Predicted Species: {species}")
