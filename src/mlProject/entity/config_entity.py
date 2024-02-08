from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir : Path
    source_file : str
    save_file: str

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir:Path
    STATUS_FILE : str
    data_file :Path
    all_schema : dict



@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir:Path
    data_file :Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir:Path
    y_train_data_path: str
    x_train_data_path: str
    model_file: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir:Path
    x_test_data_path :Path
    y_test_data_path :Path
    model_file: Path
    metric_file_name:Path
    mlflow_uri:str