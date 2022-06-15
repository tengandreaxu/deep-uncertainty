from dataclasses import dataclass


@dataclass
class TrainingParameters:
    """Network Training Parameters

    Check: Reference [24], https://arxiv.org/pdf/1502.05336.pdf
    """

    learning_rate = 0.1
    epochs = 200

    # for adversarial example
    epsilon = 0.02
