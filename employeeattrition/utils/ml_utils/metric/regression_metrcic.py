from employeeattrition.entity.artifact_entity import RegressionMetricArtifact
from employeeattrition.exception.exception import EmployeeAttritionException
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import sys



def get_regression_score(y_true, y_pred)-> RegressionMetricArtifact:
    try:

        model_mae_score = mean_absolute_error(y_true, y_pred)
        model_mse_score = mean_squared_error(y_true, y_pred)
        model_rmse_score = np.sqrt(mean_squared_error(y_true, y_pred))
        model_r2_score = r2_score(y_true, y_pred)

        regression_metric = RegressionMetricArtifact(
            mean_absolute_error=model_mae_score,
            mean_squared_error=model_mse_score,
            rmse=model_rmse_score,
            r2_score=model_r2_score
        )
        return regression_metric
    except Exception as e:
        raise EmployeeAttritionException(e, sys)
