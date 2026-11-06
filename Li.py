import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                   RidgeCV, LassoCV, ElasticNetCV, lasso_path)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Data Loader
housing = fetch_california_housing(as_frame=True)
df = housing.frame
x = housing.data
y = housing.target

# Feature engineering
x['rooms_per_household'] = x['AveRooms'] / x['HouseAge'].clip(1)
x['bedrooms_ratio'] = x['AveBedrms'] / x['AveRooms'].clip(0.1)
x['pop_per_househld'] = x['Population'] / x['AveOccup'].clip(0.1)
x['income_sqrd'] = x['MedInc'] ** 2

print(f"Dataset shape: {x.shape}")
print(f"Features: {list(x.columns)}")
print(df.describe())

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.transform(x_test)

# Model evaluation function
def evaluate_model(name, model, x_tr, x_te, y_tr, y_te):
    model.fit(x_tr, y_tr)
    y_pred = model.predict(x_te)
    y_pred_tr = model.predict(x_tr)
    results = {
        'model': name,
        'train_r2': r2_score(y_tr, y_pred_tr),
        'test_r2': r2_score(y_te, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_te, y_pred)),
        'mae': mean_absolute_error(y_te, y_pred),
        'n_nonzero': np.sum(model.coef_ != 0)
    }
    return results


alpha_grid = np.logspace(-3, 4, 100)

ols   = LinearRegression()
ridge = RidgeCV(alphas=alpha_grid, cv=5, scoring='neg_mean_squared_error')
lasso = LassoCV(alphas=alpha_grid, cv=5, max_iter=10000, random_state=42)
enet  = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
                     alphas=alpha_grid, cv=5, max_iter=10000, random_state=42)

models = {'OLS': ols, 'Ridge': ridge, 'LASSO': lasso, 'ElasticNet': enet}
results = []


for name, model in models.items():
    res = evaluate_model(name, model, x_train_s, x_test_s, y_train, y_test)
    results.append(res)
    print(f"{name:12} | R2 = {res['test_r2']:.4f} | RMSE = {res['rmse']:.4f}")


print(f"\nBest alphas: Ridge = {ridge.alpha_:.4f}, LASSO = {lasso.alpha_:.4f}")
print(f"ElasticNet: alpha = {enet.alpha_:.4f}, l1_ratio = {enet.l1_ratio_:.2f}")
print(f"LASSO zeroed {np.sum(lasso.coef_ == 0)} out of {len(lasso.coef_)} features")

# Lasso path
alphas_lasso, coefs_lasso, _ = lasso_path(x_train_s, y_train,
                                           alphas=np.logspace(-4, 1, 100),
                                           max_iter=10000)

# Ridge path
ridge_coefs = []
for a in np.logspace(-4, 4, 100):
    m = Ridge(alpha=a).fit(x_train_s, y_train)
    ridge_coefs.append(m.coef_)
ridge_coefs = np.array(ridge_coefs)  #  Outside the loop

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))  #  subplots not subplot

# reverse both arrays so x-axis goes low → high alpha (consistent with Ridge)
for i, feat in enumerate(x.columns):
    axes[0].plot(np.log10(alphas_lasso[::-1]), coefs_lasso[i][::-1], label=feat)
axes[0].axvline(np.log10(lasso.alpha_), linestyle='--', color='red', label='CV optimal')
axes[0].set_title('LASSO Regularization Path')
axes[0].set_xlabel('log10(lambda)')
axes[0].legend(fontsize=8)

for i, feat in enumerate(x.columns):
    axes[1].plot(np.log10(np.logspace(-4, 4, 100)), ridge_coefs[:, i], label=feat)
axes[1].axvline(np.log10(ridge.alpha_), linestyle='--', color='red', label='CV optimal')
axes[1].set_title('Ridge Regularization Path')
axes[1].set_xlabel('log10(lambda)')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('regularization_paths.png', dpi=150)
plt.show()

# Residual diagnostics
def residual_diagnostics(model, x_te, y_te, model_name):
    y_pred = model.predict(x_te)
    residuals = y_te - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{model_name} - Residual Diagnostics', fontsize=14)

    axes[0, 0].scatter(y_pred, y_te, alpha=0.3, s=10)
    axes[0, 0].plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'r--')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_title('Predicted vs Actual')

    axes[0, 1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[0, 1].axhline(0, color='red', linestyle='--')
    axes[0, 1].set_title('Residuals vs Fitted')

    axes[1, 0].hist(residuals, bins=60, edgecolor='k', alpha=0.7)
    axes[1, 0].set_title('Residual Distribution')

    feat_names = list(x.columns)
    coef_df = pd.Series(model.coef_, index=feat_names).sort_values()
    coef_df.plot(kind='barh', ax=axes[1, 1],
                 color=['#e85b5b' if v < 0 else '#5b8dee' for v in coef_df])
    axes[1, 1].set_title('Feature Coefficients')

    plt.tight_layout()
    return fig

# Cross-validation comparison
for name, model in models.items():
    cv_scores = cross_val_score(model, x_train_s, y_train, cv=5, scoring='r2')
    print(f"{name:12} CV R2: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


for name, model in models.items():
    fig = residual_diagnostics(model, x_test_s, y_test, name)
    fig.savefig(f'residuals_{name}.png', dpi=120)
    plt.show()