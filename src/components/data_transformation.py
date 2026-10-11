import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object # We will create utils.py shortly for saving objects

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for creating the data transformation pipeline
        based on your data types (numerical and categorical).
        """
        try:
            # Numerical and categorical feature columns
            numerical_columns = ["store_nbr", "onpromotion"]
            categorical_columns = ["family"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            logging.info(f"Read train data completed. Shape: {train_df.shape}")

            # Define target and feature columns
            target_column_name = "sales"

            # Check if test_path has target column, otherwise split a validation set from train_df
            test_df = pd.read_csv(test_path)
            if target_column_name in test_df.columns:
                val_df = test_df
            else:
                logging.info(f"Target column '{target_column_name}' not in test data. Creating validation split from training data.")
                # Take last 20% or a sample for validation
                val_size = int(len(train_df) * 0.2)
                val_df = train_df.iloc[-val_size:].copy()
                train_df = train_df.iloc[:-val_size].copy()

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(columns=[target_column_name, "id", "date"], errors="ignore")
            target_feature_train_df = train_df[target_column_name]

            input_feature_val_df = val_df.drop(columns=[target_column_name, "id", "date"], errors="ignore")
            target_feature_val_df = val_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and validation dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_val_arr = preprocessing_obj.transform(input_feature_val_df)

            # Ensure arrays are dense numpy arrays
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_val_arr, "toarray"):
                input_feature_val_arr = input_feature_val_arr.toarray()

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_val_arr, np.array(target_feature_val_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)