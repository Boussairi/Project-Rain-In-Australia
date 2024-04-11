
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import StackingClassifier


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class Models :

    def get_models(self):
        """
        Renvoie un dictionnaire de modèles et une liste de tuples (nom du modèle, modèle).
        
        Entrées:
        Aucune entrée requise.

        Sorties:
        models : dict
            Un dictionnaire contenant des noms de modèles comme clés et les instances de ces modèles comme valeurs.
        model_tuples : list
            Une liste de tuples contenant des noms de modèles et les instances de ces modèles.
        """
        models = {}
        models['DT'] = DecisionTreeClassifier()
        models['RF'] = RandomForestClassifier()
        models['knn'] = KNeighborsClassifier()
        #models['xgb'] = xgboost.XGBClassifier()
        models['svm'] = SVC()
        models['naivebayes'] = GaussianNB()
        model_tuples = [(name, estimator) for name, estimator in models.items()]
        return models, model_tuples
