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

# Numerical tolerance for deciding whether a coefficient is zero
tolerance = 1e-6

# To store results across alphas and lambdas
errors_grid = np.zeros((len(alphas), CV, len(lambdas)))
coefs_grid = np.zeros((len(alphas), CV, len(lambdas), p))
bic_grid = np.zeros((len(alphas), CV, len(lambdas)))

total_iterations = CV * len(alphas) * len(lambdas)

# Training loop with overall progress bar
with tqdm(total=total_iterations, desc="Tuning Hyperparameters", unit="fit") as pbar:
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        df_train = data.iloc[train_index]
        df_val = data.iloc[test_index]
        
        X_train, y_train, fitted_preprocessor = preprocess_data(df_train)
        X_val, y_val = fitted_preprocessor.transform(df_val)
        # Ensure training data matches the baseline dimensions found in full dataset, reindex because one-hot encoding can create different columns in different folds.
        X_train = X_train.reindex(columns=X_train_full.columns, fill_value=0)
        X_val = X_val.reindex(columns=X_train_full.columns, fill_value=0)

        n_train = len(y_train)
        
        for a_idx, alpha in enumerate(alphas):
            for j, lambda_ in enumerate(lambdas):
                with warnings.catch_warnings(): 
                    warnings.simplefilter("ignore")
                    
                    model = linear_model.ElasticNet(l1_ratio=alpha, alpha=lambda_, max_iter=10000, random_state=42)
                    model.fit(X_train, y_train)
                    
                    #store coefficients and errors
                    coefs_grid[a_idx, i, j, :] = model.coef_
                    val_preds = model.predict(X_val)
                    errors_grid[a_idx, i, j] = mean_squared_error(y_val, val_preds)

                    # Approximate BIC using training RSS and number of active coefficients - ex week 2.4
                    train_preds = model.predict(X_train)
                    rss = np.sum((y_train - train_preds) ** 2)
                    rss = max(rss, 1e-12)  # avoid log(0)

                    k_active = np.sum(np.abs(model.coef_) > tolerance) + 1  # +1 for intercept
                    bic_grid[a_idx, i, j] = n_train * np.log(rss / n_train) + np.log(n_train) * k_active

                    
                pbar.update(1)

# Calculate means and standard errors over the CV folds
cv_means = np.mean(errors_grid, axis=1)
cv_ses = np.std(errors_grid, axis=1) / np.sqrt(CV)
bic_means = np.mean(bic_grid, axis=1)


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

# 3. BIC-based selection across all alphas and lambdas
best_bic_flat = np.argmin(bic_means)
bic_a_idx, bic_l_idx = np.unravel_index(best_bic_flat, bic_means.shape)

bic_alpha = alphas[bic_a_idx]
bic_lambda = lambdas[bic_l_idx]
bic_value = bic_means[bic_a_idx, bic_l_idx]

print(f"\n--- TUNING RESULTS ---")
print(f"Optimal Alpha (l1_ratio): {min_alpha}")
print(f'Lambda Min: {min_lambda:.4f} (Error: {min_err:.4f})')
print(f'Lambda 1-SE: {l_1se:.4f} (Error: {err_1se:.4f})')
print(f"BIC-selected alpha: {bic_alpha}")
print(f"BIC-selected lambda: {bic_lambda:.4f} (Mean BIC: {bic_value:.4f})")        

# Extract arrays specifically for the best alpha
best_alpha_cv_means = cv_means[best_a_idx, :]
best_alpha_cv_ses = cv_ses[best_a_idx, :]
best_alpha_bic_means = bic_means[best_a_idx, :]
best_alpha_coefs = coefs_grid[best_a_idx, :, :, :]

# Average coefficients across folds to plot feature paths
mean_coefs = np.mean(best_alpha_coefs, axis=0)  # Shape: (lambdas, p)

# BIC minimum within the best CV alpha
best_alpha_bic_idx = np.argmin(best_alpha_bic_means)
best_alpha_bic_lambda = lambdas[best_alpha_bic_idx]

# # --- VISUALIZATIONS ---

# # PLOT 1: Hyperparameter Grid Landscape (All Alphas)
# plt.figure(figsize=(10, 6))
# for a_idx, alpha in enumerate(alphas):
#     # Highlight the best alpha with a thicker line
#     lw = 2.5 if alpha == min_alpha else 1.0
#     plt.semilogx(lambdas, cv_means[a_idx, :], label=f'Alpha = {alpha}', linewidth=lw)
# plt.xlabel('Complexity Penalty (Lambda)', fontsize=12)
# plt.ylabel('Mean Squared Error', fontsize=12)
# plt.title('CV MSE vs. Lambda for Tested Alphas (l1_ratio)', fontsize=14, fontweight='bold')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

# PLOT 2: One-SE Rule for Optimal Alpha
plt.figure(figsize=(12, 5))
plt.errorbar(lambdas, best_alpha_cv_means, yerr=best_alpha_cv_ses, fmt='o-', 
                color=DTU_RED, ecolor='lightgray', capsize=3, label='CV Mean Error')
plt.axhline(threshold, color=DTU_NAVY, linestyle='--', label='One-SE Threshold')
plt.axvline(min_lambda, color='gray', linestyle=':', label='Min $\\lambda$')
plt.axvline(l_1se, color=DTU_NAVY, linestyle='-', label='Selected (One-SE)')
plt.xscale('log')
plt.xlabel('Complexity Penalty ($\\lambda$)', color=DTU_NAVY, fontsize=12)
plt.ylabel('Mean Squared Error', color=DTU_NAVY, fontsize=12)
plt.title(f'The One-SE Rule ($\\alpha$ = {min_alpha})', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# PLOT 3: Coefficient Profile (Trace Plot) with highlighted retained features
plt.figure(figsize=(12, 6))

chosen_lambda_idx = idx_1se
chosen_coefs = mean_coefs[chosen_lambda_idx, :]
feature_names = preprocessor.get_feature_names_out()

# Retained features at chosen lambda
retained_idx = np.where(np.abs(chosen_coefs) > tolerance)[0]

# Sort retained features by absolute coefficient size
retained_sorted_idx = retained_idx[np.argsort(np.abs(chosen_coefs[retained_idx]))[::-1]]

# Show only top N in legend/highlight
max_legend_items = retained_idx.size  # or set to a fixed number like 10 for very high-dimensional data
top_idx = retained_sorted_idx[:max_legend_items]

legend_handles = []
legend_labels = []

for feature_idx in range(p):
    # Highlight top retained features
    if feature_idx in top_idx:
        line, = plt.semilogx(
            lambdas,
            mean_coefs[:, feature_idx],
            linewidth=2.2,
            alpha=0.95
        )
        legend_handles.append(line)
        legend_labels.append(f"{feature_names[feature_idx]} ({chosen_coefs[feature_idx]:.2f})")
    else:
        # Fade all other paths
        plt.semilogx(
            lambdas,
            mean_coefs[:, feature_idx],
            color='gray',
            linewidth=1.2,
            alpha=0.25
        )

# Selected lambda line
lambda_line = plt.axvline(
    l_1se,
    color='black',
    linestyle='--',
    linewidth=2.5
)

plt.xlabel('Complexity Penalty ($\\lambda$)', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.title(f'Elastic Net Coefficient Paths ($\\alpha$ = {min_alpha})', fontsize=16, fontweight='bold')
plt.xlim(lambdas[0], lambdas[-1])
plt.grid(True, alpha=0.3)

# Legend outside the plot
plt.legend(
    legend_handles + [lambda_line],
    legend_labels + [fr'Selected $\lambda$ (1-SE) = {l_1se:.3f}'],
    title=f'Top {min(max_legend_items, len(retained_idx))} retained features\n(coeff. at selected $\\lambda$)',
    title_fontsize=10,
    fontsize=9,
    loc='upper left',
    bbox_to_anchor=(1.02, 1.0),
    frameon=True,
    handlelength=2.5,
    labelspacing=0.7,
    borderpad=1
)

plt.tight_layout()
plt.show()

# PLOT 4: CV Error and BIC Comparison for Best CV Alpha
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.semilogx(
    lambdas,
    best_alpha_cv_means,
    'o-',
    color=DTU_RED,
    markersize=4,
    label='CV Mean Error'
)
ax1.axvline(min_lambda, color='gray', linestyle=':', label='Min Lambda (CV)')
ax1.axvline(l_1se, color=DTU_NAVY, linestyle='-', linewidth=2, label='Selected (One-SE)')
ax1.axvline(best_alpha_bic_lambda, color='green', linestyle='--', linewidth=2, label='Selected (BIC)')
ax1.set_xlabel('Complexity Penalty (Lambda)', fontsize=12)
ax1.set_ylabel('Mean Squared Error', color=DTU_RED, fontsize=12)
ax1.tick_params(axis='y', labelcolor=DTU_RED)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.semilogx(
    lambdas,
    best_alpha_bic_means,
    linewidth=2,
    color='green',
    label='Mean BIC'
)
ax2.set_ylabel('BIC', color='green', fontsize=12)
ax2.tick_params(axis='y', labelcolor='green')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.title(f'CV Error and BIC across Lambda (Alpha = {min_alpha})', fontsize=14, fontweight='bold')
plt.show()

# --- FEATURE SELECTION SUMMARY ---
# Find the coefficients at the chosen lambda (1-SE trick)
chosen_lambda_idx = idx_1se
chosen_coefs = mean_coefs[chosen_lambda_idx, :]

# Consider a coefficient "zero" if its absolute value is tiny (floating point safe)
#tolerance = set up earlier as 1e-6
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
    shown = 0
    for idx in top_indices:
        if np.abs(chosen_coefs[idx]) > tolerance:
            shown += 1
            print(f"  {shown}. {feature_names[idx]}: {chosen_coefs[idx]:.4f}")
            if shown == min(5, non_zero_features):
                break

# print('\nVERDICT:')
# print(f'The auditor selects alpha = {min_alpha} and lambda = {l_1se:.4f}.')
# print('While its error is slightly higher than the minimum, it is within')
# print('one standard error, meaning the difference is likely noise.')
# print('The larger lambda results in a more parsimonious, robust model.')