
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
#import xgboost as xgb
from sklearn.metrics import classification_report




class Models :


    def __init__(self):
        pass


    def get_models(self):
        """
        Gives a dictionnary of models and a list of tuples (model_name, model estimator)

        args:
            no input variable

        Returns:
            models : a dictionnary of models along with their associated estimators
            model_tuples : a list of tuples of models along with their associated estimators
        """
        models = {}
        models['DT'] = DecisionTreeClassifier()
        models['RF'] = RandomForestClassifier()
        models['knn'] = KNeighborsClassifier()
        #models['xgb'] = xgb.XGBClassifier()
        models['svm'] = SVC()
        models['naivebayes'] = GaussianNB()
        model_tuples = [(name, estimator) for name, estimator in models.items()]
        return models, model_tuples
    

    def get_stacking(self, models_tuple):
        """
        Gives a stacking model based on the input models

        args:
            models_tuple : a list of tuples of models along with their associated estimators

        Returns: 
            model : Instance of the stacking model
        """
        # define our base  models
        level0 = models_tuple
        # define our meta learner model
        level1 = LogisticRegression()
        # define the stacking ensemble
        model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
        return model
       

    def hard_voting(self, models_tuple): 
        """
        Gives a hard voting model based on the input models

        args:
            models_tuple : a list of tuples of models along with their associated estimators

        Returns: 
            hard_voting_model : Instance of the hard voting model
        """
        hard_voting_model = VotingClassifier(estimators= models_tuple, voting='hard')
        return hard_voting_model
    

    def soft_voting(self):
        """
        Gives a soft voting model based on the input models

        args: 
            no input required
        Returns: 
            soft_voting_model : Instance of the soft voting model
        """
        list_tuples = [('DT', DecisionTreeClassifier(max_leaf_nodes=10)),
                    ('RF', RandomForestClassifier()),
                    ('knn', KNeighborsClassifier()),
                    ('svm', SVC(probability=True)),
                    ('naivebayes', GaussianNB())]
        soft_voting_model = VotingClassifier(estimators= list_tuples, voting='soft')

        return soft_voting_model
        

    def fit_model(self,model, X_train, y_train): 
        """
        fits the model to the training data

        args: 
            X_train: our training features
            y_train: our training target
        Returns: 
            No output
        """
        model.fit(X_train, y_train)

    def get_predictions(self, model, X_test):
        """
        Gives predictions using the model provided

        args: 
            model: the trained model
            X_test: the data we want to test on (features)
        Returns: 
            predictions : predicted target using the model given
        """ 
        predictions = model.predict(X_test)
        return predictions
    
    
    def evaluate_model_metrics(self, y_test, predictions):
        """
        Evaluate the model using classification report.

        args:
            y_test: the true test target
            predictions: the predicted target
           
        Returns:
            report_df : dataframe containing the report of evaluation metrics (without accuracy)
            accuracy : the accuracy value extracted from the report dictionnary
                
        """
        report = classification_report(y_test, predictions,output_dict=True)
        accuracy = report.pop('accuracy', report['accuracy'])

        report_df = pd.DataFrame(report).transpose()
        
        return report_df, accuracy