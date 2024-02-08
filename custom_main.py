from mlProject import logger
from mlProject.pipeline.custom_stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlProject.pipeline.custom_stage_02_data_validation import DataValidationTrainingPipeline
from mlProject.pipeline.custom_stage_03_data_transformation import DataTransformationTrainingPipeline

from mlProject.pipeline.custom_stage_04_model_trainer   import ModelTrainerTrainingPipeline
from mlProject.pipeline.custom_stage_05_model_evaluation  import ModelEvaluationTrainingPipeline

STAGE_NAME ='Data Ingestion stage'


if __name__ == '__main__':
    try:
        logger.info(f'>>>> stage {STAGE_NAME} started <<<<<<')
        data_ingestion = DataIngestionTrainingPipeline()
        data_ingestion.main()
        logger.info(f">>>>> stage {STAGE_NAME} complated <<<<<\n\nx======x")
    except Exception as e:
        logger.exception(e)
        raise e
    


STAGE_NAME ='Data Validation stage'


if __name__ == '__main__':
    try:
        logger.info(f'>>>> stage {STAGE_NAME} started <<<<<<')
        data_validation =DataValidationTrainingPipeline()
        data_validation.main()
        logger.info(f">>>>> stage {STAGE_NAME} complated <<<<<\n\nx======x")
    except Exception as e:
        logger.exception(e)
        raise e
    


STAGE_NAME ='Data Transformation stage'


if __name__ == '__main__':
    try:
        logger.info(f'>>>> stage {STAGE_NAME} started <<<<<<')
        data_transformation =DataTransformationTrainingPipeline()
        data_transformation.main()
        logger.info(f">>>>> stage {STAGE_NAME} complated <<<<<\n\nx======x")
    except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME ='Model Trainer stage'


if __name__ == '__main__':
    try:
        logger.info(f'>>>> stage {STAGE_NAME} started <<<<<<')
        model_trainer = ModelTrainerTrainingPipeline()
        model_trainer.main()
        logger.info(f">>>>> stage {STAGE_NAME} complated <<<<<\n\nx======x")
    except Exception as e:
        logger.exception(e)
        raise e
    

STAGE_NAME ='Model Evaluation stage'


if __name__ == '__main__':
    try:
        logger.info(f'>>>> stage {STAGE_NAME} started <<<<<<')
        data_evaluation=ModelEvaluationTrainingPipeline()
        data_evaluation.main()
        logger.info(f">>>>> stage {STAGE_NAME} complated <<<<<\n\nx======x")
    except Exception as e:
        logger.exception(e)
        raise e