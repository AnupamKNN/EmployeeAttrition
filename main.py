from employeeattrition.components.data_ingestion import DataIngestion
from employeeattrition.components.data_validation import DataValidation
from employeeattrition.exception.exception import EmployeeAttritionException
from employeeattrition.logging.logger import logging
from employeeattrition.entity.config_entity import DataIngestionConfig, DataValidationConfig
from employeeattrition.entity.config_entity import TrainingPipelineConfig

import sys
if __name__ == "__main__":
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(training_pipeline_config= trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact = data_ingestion.initia_data_ingestion()
        logging.info("Data Ingestion completed")
        print(dataingestionartifact)
        data_validation_config = DataValidationConfig(training_pipeline_config= trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact, data_validation_config)
        logging.info("Initiate the data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed")
        print(data_validation_artifact)

        

    except Exception as e:
        raise EmployeeAttritionException(e, sys)