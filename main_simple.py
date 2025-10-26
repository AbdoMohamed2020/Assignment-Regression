"""
A simplified, educational implementation of Linear Regression using Stochastic Gradient Descent (SGD).
This script focuses on the core algorithm without extra features.
"""

import csv
import random

def load_dataset(file_path):
    """
    Loads features and targets from a CSV file.
    """
    feature_data = []
    target_data = []
    with open(file_path, 'r') as file_handle:
        csv_reader = csv.reader(file_handle)
        for data_row in csv_reader:
            if len(data_row) >= 4:
                feature_data.append([float(value) for value in data_row[:3]])
                target_data.append(float(data_row[3]))
    return feature_data, target_data

def normalize_features(features):
    """
    Scales feature values to a [0, 1] range using Min-Max normalization.
    """
    num_features = len(features[0])
    min_values = [min(sample[j] for sample in features) for j in range(num_features)]
    max_values = [max(sample[j] for sample in features) for j in range(num_features)]
    
    scaled_features = []
    for sample in features:
        scaled_row = []
        for j in range(num_features):
            if (max_values[j] - min_values[j]) != 0:
                scaled_value = (sample[j] - min_values[j]) / (max_values[j] - min_values[j])
                scaled_row.append(scaled_value)
            else:
                scaled_row.append(0)
        scaled_features.append(scaled_row)
    return scaled_features, min_values, max_values

class BasicLinearRegressor:
    """
    A simple Linear Regression model trained using Stochastic Gradient Descent.
    """
    def __init__(self, learning_rate=0.01, training_epochs=1000):
        self.alpha = learning_rate
        self.epochs = training_epochs
        self.model_weights = None
        self.model_bias = None
    
    def fit(self, training_features, training_targets):
        """
        Trains the model using the provided training data.
        """
        random.seed(42)
        num_samples, num_features = len(training_features), len(training_features[0])
        
        self.model_weights = [random.uniform(-0.1, 0.1) for _ in range(num_features)]
        self.model_bias = 0.0
        
        for epoch in range(self.epochs):
            shuffled_indices = list(range(num_samples))
            random.shuffle(shuffled_indices)
            
            for index in shuffled_indices:
                current_features = training_features[index]
                current_target = training_targets[index]
                
                prediction = self.model_bias + sum(self.model_weights[i] * current_features[i] for i in range(num_features))
                
                prediction_error = prediction - current_target
                
                for j in range(num_features):
                    self.model_weights[j] -= self.alpha * prediction_error * current_features[j]
                self.model_bias -= self.alpha * prediction_error
            
            if (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs} completed.")
    
    def predict(self, feature_set):
        """
        Generates predictions for a given set of features.
        """
        all_predictions = []
        for features in feature_set:
            prediction = self.model_bias + sum(self.model_weights[i] * features[i] for i in range(len(features)))
            all_predictions.append(prediction)
        return all_predictions
    
    def compute_r2(self, features, targets):
        """
        Calculates the R-squared score to evaluate model performance.
        """
        predictions = self.predict(features)
        target_mean = sum(targets) / len(targets)
        
        total_sum_squares = sum((target - target_mean) ** 2 for target in targets)
        residual_sum_squares = sum((target - pred) ** 2 for target, pred in zip(targets, predictions))
        
        r2_score = 1 - (residual_sum_squares / total_sum_squares) if total_sum_squares != 0 else 0
        return r2_score

def main():
    """
    Main function to execute the simple linear regression demonstration.
    """
    print("--- Simple Linear Regression with SGD ---")
    
    features, targets = load_dataset("Dataset.csv")
    print(f"Loaded {len(features)} data samples.")
    
    scaled_features, _, _ = normalize_features(features)
    print("Feature scaling complete.")
    
    print("\nInitiating model training...")
    regressor = BasicLinearRegressor(learning_rate=0.01, training_epochs=1000)
    regressor.fit(scaled_features, targets)
    
    r2_performance = regressor.compute_r2(scaled_features, targets)
    print(f"\nModel R-squared Performance: {r2_performance:.4f}")
    print(f"Final Weights: {[f'{weight:.4f}' for weight in regressor.model_weights]}")
    print(f"Final Bias: {regressor.model_bias:.4f}")
    
    final_predictions = regressor.predict(scaled_features)
    print(f"\n--- Sample Predictions ---")
    for i in range(5):
        prediction_error = abs(targets[i] - final_predictions[i])
        print(f"Sample {i+1}: Actual={targets[i]:.2f}, Predicted={final_predictions[i]:.2f}, Error={prediction_error:.2f}")

if __name__ == "__main__":
    main()
