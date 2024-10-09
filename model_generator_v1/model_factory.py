from model_generator_v1.basic_models import SigmoidModel
# from model_generator.ode_models import LogisticModel
from model_generator_v1.ode_models import ACPModel

class ModelFactory:
    @staticmethod
    def create_model(model_type: str, **params):
        """
        Factory method to create a model based on the model_type string.
        
        Parameters:
        - model_type (str): The type of the model ('sigmoid', 'logistic', 'acp', etc.).
        - params (dict): Parameters for the model.
        
        Returns:
        - An instance of the selected model.
        """
        if model_type == 'sigmoid':
            return SigmoidModel(**params)
        elif model_type == 'acp':
            return ACPModel(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
