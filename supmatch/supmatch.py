from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class SupmatchRegressor(RegressorMixin,BaseEstimator):
    def __init__(self,sample=0.5,mtry=1,nrounds=100,seed=None):
        self.sample = sample
        self.mtry = mtry
        self.nrounds = nrounds
        self.seed = seed


    def fit_pandas(self,data: pd.DataFrame, label):
        if not isinstance(data,pd.DataFrame):
            raise TypeError("Data must be Pandas DataFrame")
        
        y = data[label]
        X = data[[col for col in data.columns if col != label]]
        return self.fit(X,y)


        
    
    def fit(self,X,y):
        if isinstance(X,pd.DataFrame):
            X = X.sort_index(axis=1)
            X = np.array(X)
        if isinstance(y, (pd.DataFrame,pd.Series)):
            y = np.array(y)

        X,y=check_X_y(X,y)

        X = (X-X.mean(axis=0)) / X.std(axis=0)
        self._scaled_X = X
        self._y = y

        seed = self.seed
        sample = self.sample
        nrounds = self.nrounds
        mtry = self.mtry
  
        if seed is None:
            seed = 0
        rng = np.random.default_rng(seed=seed)
        x_columns = np.arange(X.shape[1])
        x_rows = np.arange(X.shape[0])
        size = round(len(x_rows) * sample)
        mtry_size = round(len(x_columns) * mtry)
        out = []
        for _ in range(nrounds):
            row_idx = rng.choice(x_rows,size=size,replace=False)
            col_idx = rng.choice(x_columns,size=mtry_size,replace=False)
            row_idx.sort()
            col_idx.sort()
            out.append((row_idx,col_idx))

        self._model_list = out
        self.fitted_ = True

        return self
    
    def get_model_list(self):
        return self._model_list
    
    def get_x(self):
        return self._scaled_X
    
    def get_y(self):
        return self._y

    def _more_tags(self):
        return {
            "poor_score":True
        }
    
    def _predict_inner(self,model: tuple[np.array,np.array],newx: np.array):
        ## X = NxK  newx = MxK, relation matrix = NxM
        ## The relation matrix represents the cosine similarity between
        ## newx and model. Cosine similarity is the matrix product
        ## of X.T@newx divided by the outer product of the magnitude 
        ## of the rows of X with the magnitude of the columns of newx.
        ## The idea is that Eij of the relation matrix will be the cosine
        ## similarity of row i of X with row j of newx.

        X,y = self.get_x(), self.get_y()
        row_idx,col_idx = model
        locator = np.ix_(row_idx,col_idx)
        newx = newx[:,col_idx]
        X = X[locator]
        y = y[row_idx]
        matprod = X@newx.T # NxM
        magnitude_x_rows = np.linalg.norm(X,axis=1) # Nx1
        magnitude_newx_cols = np.linalg.norm(newx.T,axis=0) # Mx1
        elemprod = np.outer(magnitude_x_rows,magnitude_newx_cols) # NxM
        relation_matrix = matprod / elemprod # NxM
        best_idx = np.argmax(relation_matrix,axis=0) # Mx1
        return y[best_idx] # Mx1
        
        
    

    def predict(self,newx: np.array | pd.DataFrame,verbose=False,return_all=False):
        check_is_fitted(self)
        if isinstance(newx,pd.DataFrame):
            newx = newx.sort_index(axis=1)
            newx = np.array(newx)

        newx = check_array(newx)
        normnewx = (newx-newx.mean(axis=0)) / newx.std(axis=0)
        for i,model in enumerate(self.get_model_list()):
            if verbose:
                print("Predicting Round ", i)
            if i == 0:
                preds = self._predict_inner(model,normnewx)
            else:
                preds = np.concatenate((preds,self._predict_inner(model,normnewx)))

        preds = preds.reshape(i+1,newx.shape[0]).transpose()
        if return_all:
            return preds
        else:
            return np.mean(preds,axis=1)


    

        






def test():
    from sklearn.utils.estimator_checks import check_estimator

    gen=check_estimator(SupmatchRegressor(),generate_only=True)
    for elem in gen:
        print(elem)
        try:
            elem[1](elem[0])
        except Exception as e:
            print(e)

    dta = pd.DataFrame({
        'a':[1,2,3,4,5,6],
        'b':[1,3,5,7,9,11],
        'c':[4,7,100,102,120,130],
        'd':[1,2,4,5,7,8],
        'y':[4,5,3,5,12,24],
    })



    trained = SupmatchRegressor(mtry=1,nrounds=2,sample=1).fit_pandas(dta,label="y",)
    preds  = trained.predict(dta[['a','b','c','d']],verbose=True,return_all=False)
    print(preds)
    error = np.sqrt(np.mean((preds - dta['y'])**2))
    print(error)


if __name__ == '__main__':
    test()