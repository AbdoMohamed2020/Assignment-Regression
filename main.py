"""
Linear Regression with Stochastic Gradient Descent (SGD)
Complete ML Pipeline Implementation
"""

import csv
import random
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# CONFIGURATION SETTINGS
# =============================================================================
ALPHA = 0.01
EPOCHS = 1000
TEST_SET_RATIO = 0.2
REPRODUCIBILITY_SEED = 42
ENABLE_VISUALIZATIONS = True
LOG_FREQUENCY = 100

# ANSI Color Codes
class TerminalColors:
    RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = '\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[95m', '\033[96m', '\033[97m'
    BOLD, RESET = '\033[1m', '\033[0m'
    HEADING, OK, WARN, FAIL, INFO, ACCENT = BOLD + CYAN, BOLD + GREEN, BOLD + YELLOW, BOLD + RED, BOLD + BLUE, BOLD + MAGENTA

# =============================================================================
# DATA HANDLING
# =============================================================================
def load_dataset(file_path):
    """Loads feature and target data from a CSV file."""
    feature_data, target_data = [], []
    with open(file_path, 'r') as file_handle:
        csv_reader = csv.reader(file_handle)
        for data_row in csv_reader:
            if len(data_row) >= 4:
                feature_data.append([float(value) for value in data_row[:3]])
                target_data.append(float(data_row[3]))
    return feature_data, target_data

def normalize_features(feature_set):
    """Scales features to a [0, 1] range."""
    num_features = len(feature_set[0])
    min_vals = [min(col) for col in zip(*feature_set)]
    max_vals = [max(col) for col in zip(*feature_set)]
    
    scaled_set = []
    for sample in feature_set:
        scaled_sample = [(sample[j] - min_vals[j]) / (max_vals[j] - min_vals[j]) if (max_vals[j] - min_vals[j]) != 0 else 0 for j in range(num_features)]
        scaled_set.append(scaled_sample)
    
    return scaled_set, min_vals, max_vals

def split_data(features, targets, test_ratio=TEST_SET_RATIO, seed=REPRODUCIBILITY_SEED):
    """Splits data into training and testing partitions."""
    random.seed(seed)
    num_samples = len(features)
    all_indices = list(range(num_samples))
    random.shuffle(all_indices)
    
    split_point = int(num_samples * (1 - test_ratio))
    train_indices, test_indices = all_indices[:split_point], all_indices[split_point:]
    
    train_features = [features[i] for i in train_indices]
    test_features = [features[i] for i in test_indices]
    train_targets = [targets[i] for i in train_indices]
    test_targets = [targets[i] for i in test_indices]
    
    return train_features, test_features, train_targets, test_targets

# =============================================================================
# MODEL DEFINITION
# =============================================================================
class LinearRegressorSGD:
    def __init__(self, learning_rate=ALPHA, epochs=EPOCHS, seed=REPRODUCIBILITY_SEED):
        self.alpha = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, features, targets):
        random.seed(self.seed)
        num_samples, num_features = len(features), len(features[0])
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(num_features)]
        self.bias = 0.0
        
        for i in range(self.epochs):
            epoch_loss = 0
            shuffled_indices = list(range(num_samples))
            random.shuffle(shuffled_indices)
            
            for index in shuffled_indices:
                prediction = self._predict_single_instance(features[index])
                error = prediction - targets[index]
                epoch_loss += error ** 2
                
                for j in range(num_features):
                    self.weights[j] -= self.alpha * error * features[index][j]
                self.bias -= self.alpha * error
            
            self.loss_history.append(epoch_loss / num_samples)
            
            if (i + 1) % LOG_FREQUENCY == 0:
                print(f"{TerminalColors.INFO}Epoch {i + 1}/{self.epochs}{TerminalColors.RESET}, {TerminalColors.OK}Loss: {self.loss_history[-1]:.4f}{TerminalColors.RESET}")
    
    def _predict_single_instance(self, features):
        return self.bias + sum(self.weights[i] * features[i] for i in range(len(features)))
    
    def predict(self, features):
        return [self._predict_single_instance(f) for f in features]
    
    def compute_r2(self, features, targets):
        predictions = self.predict(features)
        target_mean = sum(targets) / len(targets)
        total_variance = sum((t - target_mean) ** 2 for t in targets)
        residual_variance = sum((t - p) ** 2 for t, p in zip(targets, predictions))
        return 1 - (residual_variance / total_variance) if total_variance != 0 else 0
    
    def compute_mse(self, features, targets):
        predictions = self.predict(features)
        return sum((t - p) ** 2 for t, p in zip(targets, predictions)) / len(targets)
    
    def compute_mae(self, features, targets):
        predictions = self.predict(features)
        return sum(abs(t - p) for t, p in zip(targets, predictions)) / len(targets)

# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================
def train_model(train_features, train_targets):
    print(f"\n{TerminalColors.INFO}[4]{TerminalColors.RESET} {TerminalColors.BOLD}Training Model...{TerminalColors.RESET}")
    model = LinearRegressorSGD()
    model.fit(train_features, train_targets)
    print(f"    {TerminalColors.OK}✓ Training complete.{TerminalColors.RESET}")
    return model

def evaluate_model(model, train_feats, train_tgts, test_feats, test_tgts):
    print(f"\n{TerminalColors.HEADING}{'=' * 60}\n{TerminalColors.HEADING}{TerminalColors.BOLD}PERFORMANCE REPORT\n{TerminalColors.HEADING}{'=' * 60}{TerminalColors.RESET}")
    
    print(f"\n{TerminalColors.ACCENT}Model Parameters{TerminalColors.RESET}")
    print(f"  {TerminalColors.BOLD}Weights:{TerminalColors.RESET} {TerminalColors.CYAN}{[f'{w:.4f}' for w in model.weights]}{TerminalColors.RESET}")
    print(f"  {TerminalColors.BOLD}Bias:{TerminalColors.RESET} {TerminalColors.CYAN}{model.bias:.4f}{TerminalColors.RESET}")
    
    train_r2, train_mse, train_mae = model.compute_r2(train_feats, train_tgts), model.compute_mse(train_feats, train_tgts), model.compute_mae(train_feats, train_tgts)
    test_r2, test_mse, test_mae = model.compute_r2(test_feats, test_tgts), model.compute_mse(test_feats, test_tgts), model.compute_mae(test_feats, test_tgts)
    
    print(f"\n{TerminalColors.ACCENT}Training Metrics{TerminalColors.RESET}")
    print(f"  {TerminalColors.BOLD}R²:{TerminalColors.RESET} {TerminalColors.OK}{train_r2:.4f}{TerminalColors.RESET}, {TerminalColors.BOLD}MSE:{TerminalColors.RESET} {TerminalColors.WARN}{train_mse:.4f}{TerminalColors.RESET}, {TerminalColors.BOLD}MAE:{TerminalColors.RESET} {TerminalColors.WARN}{train_mae:.4f}{TerminalColors.RESET}")
    
    print(f"\n{TerminalColors.ACCENT}Testing Metrics{TerminalColors.RESET}")
    print(f"  {TerminalColors.BOLD}R²:{TerminalColors.RESET} {TerminalColors.OK}{test_r2:.4f}{TerminalColors.RESET}, {TerminalColors.BOLD}MSE:{TerminalColors.RESET} {TerminalColors.WARN}{test_mse:.4f}{TerminalColors.RESET}, {TerminalColors.BOLD}MAE:{TerminalColors.RESET} {TerminalColors.WARN}{test_mae:.4f}{TerminalColors.RESET}")
    
    print(f"\n{TerminalColors.ACCENT}Test Set Predictions{TerminalColors.RESET}")
    test_predictions = model.predict(test_feats)
    for i in range(min(5, len(test_feats))):
        err = abs(test_tgts[i] - test_predictions[i])
        err_color = TerminalColors.OK if err < 1.0 else TerminalColors.WARN if err < 2.0 else TerminalColors.FAIL
        print(f"  {TerminalColors.BOLD}Sample {i+1}:{TerminalColors.RESET} {TerminalColors.CYAN}Actual = {test_tgts[i]:.2f}{TerminalColors.RESET}, {TerminalColors.MAGENTA}Predicted = {test_predictions[i]:.2f}{TerminalColors.RESET}, {err_color}Error = {err:.2f}{TerminalColors.RESET}")

def optimize_hyperparameters(train_feats, train_tgts, test_feats, test_tgts):
    print(f"\n{TerminalColors.ACCENT}Optimizing Learning Rate{TerminalColors.RESET}")
    best_score, best_alpha = -float('inf'), None
    
    for alpha_candidate in [0.001, 0.01, 0.1]:
        temp_model = LinearRegressorSGD(learning_rate=alpha_candidate, epochs=500, seed=42)
        temp_model.fit(train_feats, train_tgts)
        score = temp_model.compute_r2(test_feats, test_tgts)
        print(f"  {TerminalColors.BOLD}LR: {alpha_candidate}{TerminalColors.RESET}, {TerminalColors.CYAN}Test R²: {score:.4f}{TerminalColors.RESET}")
        if score > best_score: best_score, best_alpha = score, alpha_candidate
            
    print(f"  {TerminalColors.OK}✓ Best LR: {best_alpha} (R²: {best_score:.4f}){TerminalColors.RESET}")
    return best_alpha

def create_dashboard(model, train_feats, train_tgts, test_feats, test_tgts):
    if not ENABLE_VISUALIZATIONS: return
    print(f"\n{TerminalColors.ACCENT}Generating Analysis Dashboard...{TerminalColors.RESET}")
    
    try:
        train_preds = model.predict(train_feats)
        test_preds = model.predict(test_feats)
        
        fig, axs = plt.subplots(3, 3, figsize=(22, 18))
        fig.suptitle('Gradient Descent Regression - Full Analysis', fontsize=20, fontweight='bold')

        # 1. Training Loss Convergence
        axs[0, 0].plot(model.loss_history, color='#2E86AB', linewidth=2)
        axs[0, 0].set_title('1. Training Loss Convergence', fontsize=14, fontweight='bold')
        axs[0, 0].set_xlabel('Epoch', fontweight='bold')
        axs[0, 0].set_ylabel('Loss (MSE)', fontweight='bold')
        axs[0, 0].grid(True, alpha=0.3)

        # 2. Training Set: Predictions vs Actual
        axs[0, 1].scatter(train_tgts, train_preds, alpha=0.7, color='#2E86AB')
        min_val_train = min(min(train_tgts), min(train_preds))
        max_val_train = max(max(train_tgts), max(train_preds))
        axs[0, 1].plot([min_val_train, max_val_train], [min_val_train, max_val_train], 'r--', lw=2)
        axs[0, 1].set_title('2. Training Set: Predictions vs Actual', fontsize=14, fontweight='bold')
        axs[0, 1].set_xlabel('Actual Values', fontweight='bold')
        axs[0, 1].set_ylabel('Predicted Values', fontweight='bold')
        axs[0, 1].grid(True, alpha=0.3)

        # 3. Test Set: Predictions vs Actual
        axs[0, 2].scatter(test_tgts, test_preds, alpha=0.8, color='#E63946')
        min_val_test = min(min(test_tgts), min(test_preds))
        max_val_test = max(max(test_tgts), max(test_preds))
        axs[0, 2].plot([min_val_test, max_val_test], [min_val_test, max_val_test], 'r--', lw=2)
        axs[0, 2].set_title('3. Test Set: Predictions vs Actual', fontsize=14, fontweight='bold')
        axs[0, 2].set_xlabel('Actual Values', fontweight='bold')
        axs[0, 2].set_ylabel('Predicted Values', fontweight='bold')
        axs[0, 2].grid(True, alpha=0.3)
        
        # 4. Residual Analysis
        residuals = [actual - pred for actual, pred in zip(test_tgts, test_preds)]
        axs[1, 0].scatter(test_preds, residuals, alpha=0.7, color='#F18F01')
        axs[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=2)
        axs[1, 0].set_title('4. Residual Analysis', fontsize=14, fontweight='bold')
        axs[1, 0].set_xlabel('Predicted Values', fontweight='bold')
        axs[1, 0].set_ylabel('Residuals', fontweight='bold')
        axs[1, 0].grid(True, alpha=0.3)
        
        # 5. Feature Importance
        feature_names = ['Feature 1', 'Feature 2', 'Feature 3']
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        axs[1, 1].bar(feature_names, model.weights, color=colors, alpha=0.8)
        axs[1, 1].set_title('5. Feature Importance (Weights)', fontsize=14, fontweight='bold')
        axs[1, 1].set_ylabel('Weight Value', fontweight='bold')
        axs[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 6. Error Distribution
        errors = [abs(actual - pred) for actual, pred in zip(test_tgts, test_preds)]
        axs[1, 2].hist(errors, bins=8, alpha=0.7, color='#A23B72', edgecolor='white')
        axs[1, 2].axvline(np.mean(errors), color='red', linestyle='--', lw=2, label=f'Mean Error: {np.mean(errors):.2f}')
        axs[1, 2].set_title('6. Error Distribution (Test Set)', fontsize=14, fontweight='bold')
        axs[1, 2].set_xlabel('Absolute Error', fontweight='bold')
        axs[1, 2].set_ylabel('Frequency', fontweight='bold')
        axs[1, 2].legend()
        axs[1, 2].grid(True, alpha=0.3, axis='y')
        
        # 7. Performance Metrics Comparison
        metrics = ['R² Score', 'MSE', 'MAE']
        train_r2 = model.compute_r2(train_feats, train_tgts)
        test_r2 = model.compute_r2(test_feats, test_tgts)
        train_mse = model.compute_mse(train_feats, train_tgts)
        test_mse = model.compute_mse(test_feats, test_tgts)
        train_mae = model.compute_mae(train_feats, train_tgts)
        test_mae = model.compute_mae(test_feats, test_tgts)
        train_values = [train_r2, train_mse, train_mae]
        test_values = [test_r2, test_mse, test_mae]
        x = np.arange(len(metrics))
        width = 0.35
        axs[2, 0].bar(x - width/2, train_values, width, label='Training Set', color='#2E86AB')
        axs[2, 0].bar(x + width/2, test_values, width, label='Test Set', color='#E63946')
        axs[2, 0].set_title('7. Performance Metrics: Train vs Test', fontsize=14, fontweight='bold')
        axs[2, 0].set_ylabel('Values', fontweight='bold')
        axs[2, 0].set_xticks(x)
        axs[2, 0].set_xticklabels(metrics)
        axs[2, 0].legend()
        axs[2, 0].grid(True, alpha=0.3, axis='y')

        # Hide unused subplots for a cleaner look
        fig.delaxes(axs[2, 1])
        fig.delaxes(axs[2, 2])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
        print(f"  {TerminalColors.OK}✓ Dashboard generated.{TerminalColors.RESET}")
        
    except Exception as e:
        print(f"  {TerminalColors.FAIL}✗ Visualization failed: {e}{TerminalColors.RESET}")

def main():
    """Main function to run the ML pipeline."""
    print(f"{TerminalColors.HEADING}{'=' * 60}\n{TerminalColors.HEADING}{TerminalColors.BOLD}ML Pipeline: Linear Regression with SGD\n{TerminalColors.HEADING}{'=' * 60}{TerminalColors.RESET}")
    
    print(f"\n{TerminalColors.INFO}[1]{TerminalColors.RESET} {TerminalColors.BOLD}Data Loading...{TerminalColors.RESET}")
    try:
        features, targets = load_dataset("Dataset.csv")
        print(f"    {TerminalColors.OK}[OK] Imported {len(features)} samples.{TerminalColors.RESET}")
    except FileNotFoundError:
        print(f"    {TerminalColors.FAIL}✗ FATAL: 'Dataset.csv' not found.{TerminalColors.RESET}")
        return
    
    print(f"\n{TerminalColors.INFO}[2]{TerminalColors.RESET} {TerminalColors.BOLD}Data Preprocessing...{TerminalColors.RESET}")
    train_feats, test_feats, train_tgts, test_tgts = split_data(features, targets)
    print(f"    {TerminalColors.OK}✓ Data split: {len(train_feats)} train, {len(test_feats)} test.{TerminalColors.RESET}")
    
    scaled_train_feats, min_vals, max_vals = normalize_features(train_feats)
    scaled_test_feats = [ [(sample[j] - min_vals[j]) / (max_vals[j] - min_vals[j]) if (max_vals[j] - min_vals[j]) != 0 else 0 for j in range(len(sample))] for sample in test_feats]
    print(f"    {TerminalColors.OK}✓ Feature scaling applied.{TerminalColors.RESET}")
    
    print(f"\n{TerminalColors.INFO}[3]{TerminalColors.RESET} {TerminalColors.BOLD}Model Training...{TerminalColors.RESET}")
    model = train_model(scaled_train_feats, train_tgts)
    
    print(f"\n{TerminalColors.INFO}[4]{TerminalColors.RESET} {TerminalColors.BOLD}Model Evaluation...{TerminalColors.RESET}")
    evaluate_model(model, scaled_train_feats, train_tgts, scaled_test_feats, test_tgts)
    
    print(f"\n{TerminalColors.INFO}[5]{TerminalColors.RESET} {TerminalColors.BOLD}Hyperparameter Optimization...{TerminalColors.RESET}")
    optimal_alpha = optimize_hyperparameters(scaled_train_feats, train_tgts, scaled_test_feats, test_tgts)
    
    print(f"\n{TerminalColors.INFO}[6]{TerminalColors.RESET} {TerminalColors.BOLD}Generating Final Dashboard (LR={optimal_alpha})...{TerminalColors.RESET}")
    final_regressor = LinearRegressorSGD(learning_rate=optimal_alpha)
    final_regressor.fit(scaled_train_feats, train_tgts)
    create_dashboard(final_regressor, scaled_train_feats, train_tgts, scaled_test_feats, test_tgts)
    
    print(f"\n{TerminalColors.HEADING}{'=' * 60}\n{TerminalColors.OK}{TerminalColors.BOLD}Pipeline Finished Successfully.\n{TerminalColors.HEADING}{'=' * 60}{TerminalColors.RESET}")

if __name__ == "__main__":
    main()
