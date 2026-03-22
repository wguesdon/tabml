import numpy as np, pandas as pd
from tabml.models import XGBoostModel
import xgboost as xgb
print("xgboost:", xgb.__version__)

# Regression check
rng = np.random.RandomState(0)
X = pd.DataFrame(rng.randn(400, 8))
y = pd.Series(X.sum(axis=1) + rng.randn(400)*0.1)
X_tr, X_val = X.iloc[:300], X.iloc[300:]
y_tr, y_val = y.iloc[:300], y.iloc[300:]

m = XGBoostModel(params={'n_estimators': 500, 'learning_rate': 0.1, 'max_depth': 3, 'tree_method': 'hist'})
m.fit(X_tr, y_tr, X_val=X_val, y_val=y_val, early_stopping_rounds=20)
print('use_booster:', getattr(m, '_use_booster', None))
print('best_iteration:', getattr(m.model, 'best_iteration', None))

pred = m.predict(X_val)
print('pred shape:', pred.shape, 'first3:', pred[:3])
