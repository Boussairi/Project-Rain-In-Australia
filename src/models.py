
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
    


    def evaluate_model(self, model, X, y):
        """
        Évalue les performances du modèle en utilisant la validation croisée stratifiée répétée.

        Entrées:
        model : object
            Instance du modèle à évaluer.
        X : array-like
            Tableau de forme (n_samples, n_features) contenant les données d'entraînement.
        y : array-like
            Tableau de forme (n_samples,) contenant les étiquettes cibles correspondant aux échantillons dans X.

        Sortie:
        scores : array-like
            Tableau contenant les scores d'exactitude pour chaque pli de validation croisée.
        """
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        return scores


        


    def get_stacking(self, models_tuple):
        """
        Renvoie un modèle de stacking basé sur les modèles fournis.

        Entrées:
        models_tuple : list
            Une liste de tuples contenant des noms de modèles et les instances de ces modèles.

        Sortie:
        model : object
            Instance du modèle de stacking.
        """
        # define our base  models
        level0 = models_tuple
        # define our meta learner model
        level1 = LogisticRegression()
        # define the stacking ensemble
        model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
        return model
    