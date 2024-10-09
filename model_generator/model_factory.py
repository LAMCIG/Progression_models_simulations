from model_generator.sigmoid_model import SigmoidModel

class ModelFactory:
    
    @staticmethod
    def create_model(model_type: str, **params):
        if model_type == 'sigmoid':
            return SigmoidModel(**params)
        # TODO: Add rest of the models
        else:
            raise ValueError(f"Unknown model type: {model_type}")