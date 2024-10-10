from model_generator.other_models.sigmoid_model import SigmoidModel
from model_generator.ode_models.acp_model import ACPModel
from model_generator.ode_models.logistic_model import LogisticModel
from model_generator.other_models.transition_matrix_model import TransitionMatrixModel
class ModelFactory:
    
    @staticmethod
    def create_model(model_type: str, **params):
        if model_type == 'sigmoid':
            return SigmoidModel(**params)
        if model_type == 'acp':
            return ACPModel(**params)
        if model_type == 'logistic':
            return LogisticModel(**params)
        if model_type == 'transition':
            return TransitionMatrixModel(**params)
        # TODO: Add rest of the models
        else:
            raise ValueError(f"Unknown model type: {model_type}")