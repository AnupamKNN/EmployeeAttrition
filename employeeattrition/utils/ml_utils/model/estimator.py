from employeeattrition.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

import os
import sys

from employeeattrition.exception.exception import EmployeeAttritionException
from employeeattrition.logging.logger import logging


class EmployeeModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise EmployeeAttritionException(e, sys)
        
    def predict(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            return self.model.predict(x_transform)
        except Exception as e:
            raise EmployeeAttritionException(e, sys)