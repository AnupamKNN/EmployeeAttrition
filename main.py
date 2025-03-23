from employeeattrition.components.data_ingestion import DataIngestion
from employeeattrition.components.data_validation import DataValidation
from employeeattrition.components.model_trainer import ModelTrainer
from employeeattrition.components.data_transformation import DataTransformation
from employeeattrition.exception.exception import EmployeeAttritionException
from employeeattrition.logging.logger import logging
from employeeattrition.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from employeeattrition.entity.config_entity import TrainingPipelineConfig

import sys
if __name__ == "__main__":
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(training_pipeline_config= trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiated the data ingestion")
        dataingestionartifact = data_ingestion.initia_data_ingestion()
        logging.info("Data Ingestion completed")
        print(dataingestionartifact)
        data_validation_config = DataValidationConfig(training_pipeline_config= trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact, data_validation_config)
        logging.info("Initiated the data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed")
        print(data_validation_artifact)
        data_transformation_config = DataTransformationConfig(training_pipeline_config= trainingpipelineconfig)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        logging.info("Data transformation started")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed")
        print(data_transformation_artifact)

        logging.info("Model Training started")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config= trainingpipelineconfig)
        model_trainer = ModelTrainer(model_trainer_config= model_trainer_config,
                                     data_transformation_artifact= data_transformation_artifact)
        model_trainer_artifact = model_trainer.initate_model_trainer()
        logging.info("Model Training artifact created")


        

    except Exception as e:
        raise EmployeeAttritionException(e, sys)