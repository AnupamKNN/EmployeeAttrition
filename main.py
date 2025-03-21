from employeeattrition.components.data_ingestion import DataIngestion
from employeeattrition.exception.exception import EmployeeAttritionException
from employeeattrition.logging.logger import logging
from employeeattrition.entity.config_entity import DataIngestionConfig
from employeeattrition.entity.config_entity import TrainingPipelineConfig

import sys
if __name__ == "__main__":
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(training_pipeline_config= trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        data_ingestion_artifact = data_ingestion.initia_data_ingestion()
        print(data_ingestion_artifact)

    except Exception as e:
        raise EmployeeAttritionException(e, sys)