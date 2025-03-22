import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from employeeattrition.entity.config_entity import DataTransformationConfig
from employeeattrition.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact
)

from employeeattrition.exception.exception import EmployeeAttritionException
from employeeattrition.logging.logger import logging
from employeeattrition.constant.training_pipeline import TARGET_COLUMN
from employeeattrition.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from employeeattrition.utils.main_utils.utils import save_numpy_array_data, save_object
from employeeattrition.constant.training_pipeline import SCHEMA_FILE_PATH
from employeeattrition.utils.main_utils.utils import read_yaml_file


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, 
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact = data_validation_artifact
            self.data_transformation_config:DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise EmployeeAttritionException(e, sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise EmployeeAttritionException(e, sys)
        
    def get_data_transformer_object(self) -> Pipeline:
        """
        Returns:
            A Pipeline object that applies KNN Imputation, Standard Scaling, 
            and One-Hot Encoding (with first column dropped).
        """
        logging.info("Entered get_data_transformer_object method of Transformation class")
        
        try:
            # Load schema configuration
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)

            # Identify numerical and categorical features
            num_features = [list(d.keys())[0] for d in self.schema_config["numerical_columns"]]
            cat_features = [list(d.keys())[0] for d in self.schema_config["categorical_columns"]]

            # logging.info(f"Numerical Columns: {num_features}")
            # logging.info(f"Categorical Columns: {cat_features}")

            # Define numerical transformer pipeline
            numeric_transformer = Pipeline(steps=[
                ("imputer", KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)),
                ("scaler", StandardScaler())
            ])
            logging.info(f"Initialized KNNImputer and StandardScaler with parameters: {DATA_TRANSFORMATION_IMPUTER_PARAMS}")

            # Define categorical transformer
            categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

            # Combine transformers into a ColumnTransformer
            preprocessor = ColumnTransformer(transformers=[
                ("categorical", categorical_transformer, cat_features),
                ("numerical", numeric_transformer, num_features)
            ])

            # Wrap the ColumnTransformer in a pipeline
            transformation_pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor)
            ])
            
            return transformation_pipeline

        except Exception as e:
            logging.error("Error in get_data_transformer_object method")
            raise EmployeeAttritionException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Start data transformation")

            # Read training and testing data
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Extract input features and target features for training
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].values.reshape(-1, 1)

            # Extract input features and target features for testing
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].values.reshape(-1, 1)

            # Apply preprocessing
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)

            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            transformed_input_train_feature = transformed_input_train_feature.toarray() # THis is done because after transformation, the output is a sparse matrix
            transformed_input_test_feature = transformed_input_test_feature.toarray() # This is done because after transformation, the output is a sparse matrix

            print("Shape of transformed_input_train_feature:", transformed_input_train_feature.shape)
            print("Shape of target_feature_train_df:", target_feature_train_df.shape)

            print("Shape of transformed_input_test_feature:", transformed_input_test_feature.shape)
            print("Shape of target_feature_test_df:", target_feature_test_df.shape)

            print(f"The type of transformed_input_train_feature is: {type(transformed_input_train_feature)}")
            print(f"The type of target_feature_train_df is: {type(target_feature_train_df)}") 

            print(f"The type of transformed_input_test_feature is: {type(transformed_input_test_feature)}")
            print(f"The type of target_feature_test_df is: {type(target_feature_test_df)}")

            train_arr = np.c_[transformed_input_train_feature, target_feature_train_df]
            test_arr = np.c_[transformed_input_test_feature, target_feature_test_df]

            # Save transformed numpy arrays
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            # Save the preprocessing object
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            # Prepare and return artifacts
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact

        except Exception as e:
            raise EmployeeAttritionException(e, sys)


