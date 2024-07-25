# supmatch

Supmatch (Supervised Matching) is a scikit-learn compatible model for supervised learning that uses fuzzy matching to form predictions.

## Install

```python
python3 -m pip install https://github.com/jacgoldsm/supmatch
```

## Example usage: the diabetes dataset

```python
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.datasets import load_diabetes

from supmatch import SupmatchRegressor

data = load_diabetes()

X,y = data.data, data.target

TRAIN_SIZE = round(0.75 * X.shape[0])
train_idx = rng.choice(np.arange(X.shape[0]),size=TRAIN_SIZE,replace=False)

X_train = X[train_idx,:]
y_train = y[train_idx]

test_idx = np.setdiff1d(np.arange(X.shape[0]), train_idx)

X_test = X[test_idx,:]
y_test = y[test_idx]


def mse_supmatch():
    params = {
            'sample': [0.1,0.2,0.3,0.4,0.5,0.6, 0.8, 1.0],
            'mtry': [0.3,0.4,0.5,0.6,0.7, 0.8, 0.9,1.0],
            }

    supmatch = SupmatchRegressor(nrounds=600)
    folds = 5
    param_comb = 100

    skf = KFold(n_splits=folds, shuffle = True, random_state = 1001)
    random_search = RandomizedSearchCV(supmatch, param_distributions=params, n_iter=param_comb, scoring='neg_mean_squared_error', cv=skf.split(X_train,y_train), verbose=3, random_state=1001 )

    random_search.fit(X_train, y_train)

    print('\n All results:')
    print(random_search.cv_results_)
    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best root mse for %d-fold search with %d parameter combinations:' % (folds, param_comb))
    score = random_search.best_score_
    print(score)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)
    preds = random_search.predict(X_test)
    rmse = np.sqrt(np.mean((preds - y_test)**2))
    print("BEST SUPMATCH ROOT MSE FOR TEST DATA: ", rmse)


if __name__ == '__main__':
    mse_supmatch()
```

