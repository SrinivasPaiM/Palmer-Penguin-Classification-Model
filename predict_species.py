import pandas as pd
import joblib

# Load the trained model
model = joblib.load("penguin_species_model.pkl")
features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']

def predict_species(culmen_length, culmen_depth, flipper_length, body_mass):
    input_data = pd.DataFrame([[culmen_length, culmen_depth, flipper_length, body_mass]], columns=features)
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    return prediction[0], probabilities

if __name__ == "__main__":
    print("Welcome to the Penguin Species Predictor!")
    print("Enter the physical traits of the penguin to predict its species.")
    print("Type 'exit' to quit the program.\n")

    while True:
        try:
            # Get user input
            culmen_length = input("Enter Culmen Length (mm): ")
            if culmen_length.lower() == "exit":
                break
            culmen_length = float(culmen_length)

            culmen_depth = input("Enter Culmen Depth (mm): ")
            if culmen_depth.lower() == "exit":
                break
            culmen_depth = float(culmen_depth)

            flipper_length = input("Enter Flipper Length (mm): ")
            if flipper_length.lower() == "exit":
                break
            flipper_length = float(flipper_length)

            body_mass = input("Enter Body Mass (g): ")
            if body_mass.lower() == "exit":
                break
            body_mass = float(body_mass)

            # Predict species
            species, probabilities = predict_species(culmen_length, culmen_depth, flipper_length, body_mass)
            print(f"\nPredicted Species: {species}")
            print("Prediction Probabilities:")
            for i, prob in enumerate(probabilities[0]):
                print(f"  {model.classes_[i]}: {prob * 100:.2f}%")
            print("\n")

        except ValueError:
            print("Invalid input. Please enter numeric values for the traits.\n")
        except Exception as e:
            print(f"An error occurred: {e}\n")

    print("Thank you for using the Penguin Species Predictor!")
