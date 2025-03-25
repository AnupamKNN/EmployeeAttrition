import sys, os
import numpy as np

from employeeattrition.entity.config_entity import ModelTrainerConfig
from employeeattrition.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact

from employeeattrition.exception.exception import EmployeeAttritionException
from employeeattrition.logging.logger import logging

from employeeattrition.utils.main_utils.utils import save_object, load_object
from employeeattrition.utils.main_utils.utils import load_numpy_array_data, evaluate_models
from employeeattrition.utils.ml_utils.metric.regression_metrcic import get_regression_score
from employeeattrition.utils.ml_utils.model.estimator import EmployeeModel


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

import mlflow


import dagshub
dagshub.init(repo_owner='AnupamKNN', repo_name='EmployeeAttrition', mlflow=True)



class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_train_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise EmployeeAttritionException(e, sys)
        
    def track_mlflow(self, best_model, regession_metric):
        with mlflow.start_run():
            mae = regession_metric.mean_absolute_error
            mse = regession_metric.mean_squared_error
            rmse = np.sqrt(regession_metric.mean_squared_error)
            r2 = regession_metric.r2_score

            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(best_model, "model")
        

    def train_model(self, x_train, y_train, x_test, y_test)-> EmployeeModel:
        try:
            models = {
            "Linear Regression" : LinearRegression(),
            "KNeighbors" : KNeighborsRegressor(n_jobs=-1),
            "RandomForest" : RandomForestRegressor(verbose=1, n_jobs = -1),
            "AdaBoost" : AdaBoostRegressor(),
            "GradientBoosting" : GradientBoostingRegressor(verbose=1),
            "XGBoost" : XGBRegressor(n_jobs=-1,  tree_method='gpu_hist', predictor='gpu_predictor'),
            "SVR" : SVR(),
            "DecisionTree" : DecisionTreeRegressor()
            }

            param_grid = {
                    "Linear Regression": {
                        "fit_intercept": [True, False]
                    },
                    "KNeighbors": {
                        "n_neighbors": [3, 5, 7, 10],
                        # "weights": ["uniform", "distance"],
                        # "metric": ["euclidean", "manhattan"]
                    },
                    "RandomForest": {
                        "n_estimators": [100, 200, 500],
                        "max_depth": [3, 5, 7, 10, 20, 30, None],
                        # "min_samples_split": [2, 5, 10],
                        # "min_samples_leaf": [1, 2, 4]
                    },
                    "AdaBoost": {
                        "n_estimators": [50, 100, 200],
                        # "learning_rate": [0.01, 0.1, 1]
                    },
                    "GradientBoosting": {
                        "n_estimators": [100, 200, 500],
                        # "learning_rate": [0.01, 0.1, 0.2],
                        # "max_depth": [3, 5, 10]
                    },
                    "XGBoost": {
                        "n_estimators": [100, 200, 500],
                        "learning_rate": [0.01, 0.1, 0.2],
                        # "max_depth": [3, 5, 7, 10],
                        # "subsample": [0.7, 0.8, 1.0],
                        # "colsample_bytree": [0.7, 0.8, 1.0]
                    },
                    "SVR": {
                        "kernel": ["linear", "rbf", "poly"],
                        # "C": [0.1, 1, 10, 100],
                        # "gamma": ["scale", "auto"]
                    },
                    "DecisionTree": {
                        "max_depth": [5, 10, 20, None],
                        # "min_samples_split": [2, 5, 10],
                        # "min_samples_leaf": [1, 2, 4]
                    }
        }
            model_report: dict = evaluate_models(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, 
                                             models=models, param=param_grid)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get the best model name from dict
            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            y_train_pred = best_model.predict(x_train)

            regression_train_metric = get_regression_score(y_true = y_train, y_pred = y_train_pred)

            ## Track the experiments with mlflow for train metric
            self.track_mlflow(best_model, regression_train_metric)

            
            y_test_pred = best_model.predict(x_test)

            regression_test_metric = get_regression_score(y_true= y_test, y_pred= y_test_pred)

            ## Track the experiments with mlflow for test metric
            self.track_mlflow(best_model, regression_test_metric)

            preprocessor = load_object(file_path = self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_train_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            Employe_Model = EmployeeModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_train_config.trained_model_file_path, obj=EmployeeModel)

            save_object("final_model/model.pkl", best_model)

            ## Model Trainer Artifact
            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_train_config.trained_model_file_path,
                                 train_metric_artifact = regression_train_metric,
                                 test_metric_artifact = regression_test_metric
                                 )
            
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact


        except Exception as e:
            raise EmployeeAttritionException(e, sys)



    def initate_model_trainer(self)-> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading train array and test array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            model = self.train_model(x_train, y_train, x_test, y_test)


        except Exception as e:
            raise EmployeeAttritionException(e, sys)