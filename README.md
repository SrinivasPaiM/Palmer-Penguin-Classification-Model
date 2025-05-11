# Penguin Species Classification

This project provides tools to classify penguin species based on physical traits using a trained machine learning model.

## Features
1. **Model Training**:
   - Train a Random Forest Classifier to predict penguin species based on physical traits.
   - Save the trained model for future use.

2. **Interactive Prediction**:
   - Use a command-line interface to predict penguin species interactively.
   - Continuous mode for real-time predictions.

##Screanshots

![image](https://github.com/user-attachments/assets/6a086ebb-7384-4483-b49e-f2a14ae64781)

## Files
- **`train_model.py`**: Script to train the model and save it as `penguin_species_model.pkl`.
- **`predict_species.py`**: Interactive script to load the trained model and predict species based on user input.

## Usage
### 1. Train the Model
Run the `train_model.py` script to train the model and save it:
```bash
python train_model.py
```

### 2. Predict Species
Run the `predict_species.py` script for interactive predictions:
```bash
python predict_species.py
```
Follow the prompts to enter physical traits (Culmen Length, Culmen Depth, Flipper Length, Body Mass) and get predictions.

## Requirements
Install the required Python libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Dataset
The dataset used in this analysis is `penguins_lter_clean.csv`. Ensure the file is in the same directory as the scripts.

## Outputs
- Trained model: `penguin_species_model.pkl`

## Conclusion
This project enables species classification based on physical traits, providing a practical tool for penguin species identification.
