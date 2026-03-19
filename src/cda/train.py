import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn import linear_model
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
from data import preprocess_data

# DTU Colors
DTU_RED = '#990000'
DTU_NAVY = '#00213E'

data = pd.read_csv('data/case1Data.csv')

# Initial preprocess just to get data dimensions
X_train_full, y_train_full, preprocessor = preprocess_data(data)

CV = 5
kf = KFold(n_splits=CV, shuffle=True, random_state=42)

[n, p] = X_train_full.shape
lambdas = np.logspace(-4, 2, num=100)
# Array of alphas (l1_ratio) to test
alphas = np.array([0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0])

# To store results across alphas and lambdas
errors_grid = np.zeros((len(alphas), CV, len(lambdas)))
coefs_grid = np.zeros((len(alphas), CV, len(lambdas), p))

total_iterations = CV * len(alphas) * len(lambdas)

# Training loop with overall progress bar
with tqdm(total=total_iterations, desc="Tuning Hyperparameters", unit="fit") as pbar:
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        df_train = data.iloc[train_index]
        df_val = data.iloc[test_index]
        
        X_train, y_train, fitted_preprocessor = preprocess_data(df_train)
        X_val, y_val = fitted_preprocessor.transform(df_val)
        # Ensure training data matches the baseline dimensions found in full dataset
        X_train = X_train.reindex(columns=X_train_full.columns, fill_value=0)
        X_val = X_val.reindex(columns=X_train_full.columns, fill_value=0)
        
        for a_idx, alpha in enumerate(alphas):
            for j, lambda_ in enumerate(lambdas):
                with warnings.catch_warnings(): 
                    warnings.simplefilter("ignore")
                    
                    model = linear_model.ElasticNet(l1_ratio=alpha, alpha=lambda_, max_iter=10000)
                    model.fit(X_train, y_train)
                    
                    coefs_grid[a_idx, i, j, :] = model.coef_
                    preds = model.predict(X_val)
                    errors_grid[a_idx, i, j] = mean_squared_error(y_val, preds)
                    
                pbar.update(1)

# Calculate means and standard errors over the CV folds
cv_means = np.mean(errors_grid, axis=1)
cv_ses = np.std(errors_grid, axis=1) / np.sqrt(CV)

# 1. Find global minimum across all alphas and lambdas
best_idx_flat = np.argmin(cv_means)
best_a_idx, best_l_idx = np.unravel_index(best_idx_flat, cv_means.shape)

min_alpha = alphas[best_a_idx]
min_lambda = lambdas[best_l_idx]
min_err = cv_means[best_a_idx, best_l_idx]
min_se = cv_ses[best_a_idx, best_l_idx]

# 2. Apply One-SE Rule
threshold = min_err + min_se
possible_lambdas_idx = np.where(cv_means[best_a_idx, :] <= threshold)[0]
idx_1se = np.max(possible_lambdas_idx)
l_1se = lambdas[idx_1se]
err_1se = cv_means[best_a_idx, idx_1se]

print(f"\n--- TUNING RESULTS ---")
print(f"Optimal Alpha (l1_ratio): {min_alpha}")
print(f'Lambda Min: {min_lambda:.4f} (Error: {min_err:.4f})')
print(f'Lambda 1-SE: {l_1se:.4f} (Error: {err_1se:.4f})')        

# Extract arrays specifically for the best alpha
best_alpha_cv_means = cv_means[best_a_idx, :]
best_alpha_cv_ses = cv_ses[best_a_idx, :]
best_alpha_coefs = coefs_grid[best_a_idx, :, :, :]
# Average coefficients across folds to plot feature paths
mean_coefs = np.mean(best_alpha_coefs, axis=0)  # Shape: (lambdas, p)

# --- VISUALIZATIONS ---
# PLOT 1: Hyperparameter Grid Landscape (All Alphas)
plt.figure(figsize=(10, 6))
for a_idx, alpha in enumerate(alphas):
    # Highlight the best alpha with a thicker line
    lw = 2.5 if alpha == min_alpha else 1.0
    plt.semilogx(lambdas, cv_means[a_idx, :], label=f'Alpha = {alpha}', linewidth=lw)
plt.xlabel('Complexity Penalty (Lambda)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('CV MSE vs. Lambda for Tested Alphas (l1_ratio)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# PLOT 2: One-SE Rule for Optimal Alpha
plt.figure(figsize=(10, 6))
plt.errorbar(lambdas, best_alpha_cv_means, yerr=best_alpha_cv_ses, fmt='o-', 
                color=DTU_RED, ecolor='lightgray', capsize=3, label='CV Mean Error')
plt.axhline(threshold, color=DTU_NAVY, linestyle='--', label='One-SE Threshold')
plt.axvline(min_lambda, color='gray', linestyle=':', label='Min Lambda')
plt.axvline(l_1se, color=DTU_NAVY, linestyle='-', label='Selected (One-SE)')
plt.xscale('log')
plt.xlabel('Complexity Penalty (Lambda)', color=DTU_NAVY, fontsize=12)
plt.ylabel('Mean Squared Error', color=DTU_NAVY, fontsize=12)
plt.title(f'The One-SE Rule (Alpha = {min_alpha})', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# PLOT 3: Coefficient Profile (Trace Plot)
plt.figure(figsize=(10, 6))
for feature_idx in range(p):
    plt.semilogx(lambdas, mean_coefs[:, feature_idx])
plt.axvline(l_1se, color='black', linestyle='--', linewidth=2, label=f'Chosen Lambda ({l_1se:.4f})')
plt.xlabel('Complexity Penalty (Lambda)', fontsize=12)
plt.ylabel('Coefficient Magnitude', fontsize=12)
plt.title(f'Coefficient Paths against Lambda (Alpha = {min_alpha})', fontsize=14, fontweight='bold')
plt.xlim(lambdas[0], lambdas[-1])
plt.legend(['Features Paths', 'Chosen Lambda'] if p > 0 else [])
plt.grid(True, alpha=0.3)
plt.show()

# --- FEATURE SELECTION SUMMARY ---
# Find the coefficients at the chosen lambda (1-SE trick)
chosen_lambda_idx = idx_1se
chosen_coefs = mean_coefs[chosen_lambda_idx, :]

# Consider a coefficient "zero" if its absolute value is tiny (floating point safe)
tolerance = 1e-6
non_zero_features = np.sum(np.abs(chosen_coefs) > tolerance)
zero_features = p - non_zero_features

print("\n--- FEATURE SELECTION SUMMARY ---")
print(f"Total features considered: {p}")
print(f"Features eliminated (coef = 0): {zero_features}")
print(f"Features retained (coef != 0): {non_zero_features}")
print(f"Dimensionality reduction: {(zero_features/p)*100:.1f}% of features removed.")

# Optional: Print the names of the top contributing features (if you want)
feature_names = preprocessor.get_feature_names_out()
if non_zero_features > 0:
    # Get feature indices sorted by absolute magnitude (largest to smallest)
    top_indices = np.argsort(np.abs(chosen_coefs))[::-1]
    
    print("\nTop 5 most important baseline features (by average magnitude):")
    for i in range(min(5, non_zero_features)):
        idx = top_indices[i]
        print(f"  {i+1}. {feature_names[idx]}: {chosen_coefs[idx]:.4f}")

print('\nVERDICT:')
print(f'The auditor selects alpha = {min_alpha} and lambda = {l_1se:.4f}.')
print('While its error is slightly higher than the minimum, it is within')
print('one standard error, meaning the difference is likely noise.')
print('The larger lambda results in a more parsimonious, robust model.')

cv_rmse = np.sqrt(err_1se)
print(f"Estimated Generalization RMSE (from 5-Fold CV): {cv_rmse:.4f}")