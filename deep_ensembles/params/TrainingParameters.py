from dataclasses import dataclass


@dataclass
class TrainingParameters:
    """Network Training Parameters

    Check: Reference [24], https://arxiv.org/pdf/1502.05336.pdf
    """

    learning_rate = 0.1
    epochs = 40

    # for adversarial example
    epsilon = 0.02

    # for real datasets experiments
    folds = 20
    random_state = 42
