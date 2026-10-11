import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging

def save_object(file_path: str, obj):
    """
    This function saves a Python object (like a preprocessor or model) to disk using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        logging.info(f"Successfully saved object at: {file_path}")
        
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str):
    """
    This function loads a saved Python object from disk using pickle.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)