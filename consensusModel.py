from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn import neighbors
import xgboost as xgb
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from fingerprintGenerator import FragmentFingerprint

Y = pd.read_csv("data/curedData.csv", usecols=["label"]).squeeze()
X = pd.read_csv("data/trainMatrix.csv")

bitUsed = X.columns.values.astype(int)

SMARTS=np.loadtxt("data/smartsCSFP.csv", delimiter = "_", dtype=str, comments=None, usecols=[0])
CSFP=FragmentFingerprint(substructure_list=SMARTS[bitUsed])

clsf={  "RF"  : RandomForestClassifier(),
        "SVM" : svm.SVC(kernel="linear", probability=True),
        "XGB" : xgb.XGBClassifier(eval_metric='error'),
        "KNN" : neighbors.KNeighborsClassifier(),
        "ADA" : AdaBoostClassifier(),
      }
    
    
class ConsensusDevTox(BaseEstimator, ClassifierMixin):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._clsf={k:v for k,v in clsf.items() if k!="NB"}
        if "cutoffClass" in kwargs:
            self.cutoffClass=kwargs["cutoffClass"]
        else:
            self.cutoffClass=.5
        
        self.binary_only=True
    
    def fit(self, X, y=None):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        self.X_ = X
        self.y_ = y
        for c in self._clsf.values():
            c.fit(X,y)
        # Return the classifier
        return self
    
    def predict(self, X, y=None, with_tree=False):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        
        preds=self.predict_proba(X)
        
        return np.array([1 if i[1]>=self.cutoffClass else 0 for i in preds])
    
    def _getMatrixFromSmiles(self, smiles):
        _ = CSFP.transform_smiles(smiles)
        rawMatrix = pd.DataFrame.sparse.from_spmatrix(_)
        rawMatrix.columns = bitUsed
        return rawMatrix
    
    def predictFromSmiles(self, smiles):
        return self.predict ( self._getMatrixFromSmiles(smiles).values )
    
    def predict_proba(self, X, y=None, with_tree=False):
        preds=[c.predict_proba(X) for c in self._clsf.values()]
        mean=np.mean(preds, axis=0)
        
        return mean
    
    


TISBE = ConsensusDevTox()
TISBE.fit(X.values,Y.values)
