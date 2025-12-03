import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from src.utils.data_processing import (
    preprocess_training_data,
    split_tensors_by_ratio,
)
from src.models.utils.model_utils import train_model, evaluate_model
from src.utils.constants import (
    DEFAULT_NETWORK_SETTINGS,
    OPTIMIZED_CNN_NODE_SETTINGS,
    MODEL_TYPE_NODE,
    MODEL_TYPE_CNN_NODE,
    DIMENSION_TYPE_ENCODER,
    DIMENSION_TYPE_HIDDEN,
    DIMENSION_TYPE_REGRESSOR,
    DIMENSION_TYPE_CNN_NUM_KERNALS,
)


def learning_rate_sweep(
    model_class: str,
    training_data_directory: str,
    settings: dict = DEFAULT_NETWORK_SETTINGS,
):
    x, y, _ = preprocess_training_data(training_data_directory)

    (x_train, x_validation), (y_train, y_validation) = split_tensors_by_ratio(
        x, y, ratio=0.7
    )

    curr_settings = settings.copy()

    candidate_lrs = [0.1, 0.01, 0.005, 0.001, 0.0003]
    losses = []
    # for the sake of speed of training, evaluate based on 0.3 of train data
    # simply use the weights from the last epoch
    for lr in candidate_lrs:
        curr_settings["lr"] = lr
        model = train_model(
            model_class,
            "models/tunning_dummy.model",
            x_train,
            y_train,
            settings=curr_settings,
        )
        losses.append(
            # only look at rmse for evaluation of model for specific parameter value
            evaluate_model(
                x_validation,
                y_validation,
                model_class,
                model,
                None,
                settings=curr_settings,
                plot=False,
            )[0]
        )

    plt.figure(figsize=(12, 9))
    plt.plot(candidate_lrs, losses, marker="x")
    plt.xscale("log")
    plt.xlabel("Learning Rates")
    plt.ylabel("Validation RMSE")
    plt.grid(True)
    plt.show()


def hidden_dimensions_sweep(
    model_class: str,
    training_data_directory: str,
    settings: dict = DEFAULT_NETWORK_SETTINGS,
    dimension_type: str = DIMENSION_TYPE_HIDDEN,
):
    if dimension_type not in [
        DIMENSION_TYPE_HIDDEN,
        DIMENSION_TYPE_ENCODER,
        DIMENSION_TYPE_REGRESSOR,
        DIMENSION_TYPE_CNN_NUM_KERNALS,
    ]:
        raise ValueError(f"Unexpected dimension type given: {dimension_type}")
    x, y, _ = preprocess_training_data(training_data_directory)

    (x_train, x_validation), (y_train, y_validation) = split_tensors_by_ratio(
        x, y, ratio=0.7
    )

    curr_settings = settings.copy()

    candidate_hds = (
        [32, 64, 128]
        if dimension_type != DIMENSION_TYPE_CNN_NUM_KERNALS
        else [2, 3, 4]
        # else [20, 24, 28]
        # else [4, 8, 12, 16, 20]
    )
    losses = []
    for hd in candidate_hds:
        curr_settings[dimension_type] = hd
        model = train_model(
            model_class,
            "models/tunning_dummy.model",
            x_train,
            y_train,
            settings=curr_settings,
        )
        losses.append(
            evaluate_model(
                x_validation,
                y_validation,
                model_class,
                model,
                None,
                settings=curr_settings,
                plot=False,
            )[0]
        )

    plt.figure(figsize=(12, 9))
    plt.plot(candidate_hds, losses, marker="x")
    plt.xlabel(f"{dimension_type} for ODE")
    plt.ylabel("Validation RMSE")
    plt.grid(True)
    plt.show()


def dropout_rate_sweep(
    model_class: str,
    training_data_directory: str,
    settings: dict = DEFAULT_NETWORK_SETTINGS,
):
    x, y, _ = preprocess_training_data(training_data_directory)

    (x_train, x_validation), (y_train, y_validation) = split_tensors_by_ratio(
        x, y, ratio=0.7
    )

    curr_settings = settings.copy()

    # tuning dropout rate to see if overfit to training data
    # and need to remove some neurons so others dont become too generalized and overall more generalizable
    candidate_dor = [0, 0.05, 0.1, 0.2, 0.3]
    losses = (
        []
    )  # error/difference between the predicted value and the correct RUL values
    total_loss = (
        []
    )  # error/difference between the predicted value and the corresponding training data value
    for dor in candidate_dor:
        curr_settings["dropout"] = dor
        # trained_loss returns average loss per batch for last epoch
        trained_model, trained_loss = train_model(
            model_class,
            "models/tunning_dummy.model",
            x_train,
            y_train,
            settings=curr_settings,
            return_loss=True,
        )
        losses.append(
            evaluate_model(
                x_validation,
                y_validation,
                model_class,
                trained_model,
                None,
                settings=curr_settings,
                plot=False,
            )[0]
        )
        total_loss.append(trained_loss)

    plt.figure(figsize=(12, 9))
    plt.plot(candidate_dor, losses, color="red", marker="x")
    plt.plot(candidate_dor, total_loss, color="blue", marker="o")
    plt.xlabel("Dropout Rate")
    plt.ylabel("Validation/Training RMSE")
    plt.grid(True)
    plt.show()

    # general rule of thumb: if training loss << validation loss, increase dropout because overfitting, vice versa
    print(
        f"total losses from training set after 25 epochs {[round(l, 6) for l in total_loss]}"
    )


if __name__ == "__main__":
    # general order of tuning: learning rate, hidden dimensions, dropout rate
    # important to hold seed constant as then can compare between different neural networks with variations in their parameters
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    settings: dict = OPTIMIZED_CNN_NODE_SETTINGS
    settings["epochs"] = 10 # 10 epochs for tuning suffice

    dropout_rate_sweep(
        MODEL_TYPE_CNN_NODE,
        "CMAPSS/train_FD002.txt",
        settings=settings,
    )
