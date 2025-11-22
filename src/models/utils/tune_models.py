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
    MODEL_TYPE_NODE,
    DIMENSION_TYPE_ENCODER,
    DIMENSION_TYPE_HIDDEN,
    DIMENSION_TYPE_REGRESSOR,
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
    ]:
        raise ValueError(f"Unexpected dimension type given: {dimension_type}")
    x, y, _ = preprocess_training_data(training_data_directory)

    (x_train, x_validation), (y_train, y_validation) = split_tensors_by_ratio(
        x, y, ratio=0.7
    )

    curr_settings = settings.copy()

    candidate_hds = [32, 64, 128]
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
        trained_model, trained_loss = train_model(
            model_class,
            "models/tunning_dummy.model",
            x_train,
            y_train,
            settings=curr_settings,
            return_loss=True,
        )
        rmse, _ = evaluate_model(
            x_validation,
            y_validation,
            model_class,
            trained_model,
            None,
            settings=curr_settings,
            plot=False,
        )
        losses.append(rmse)
        total_loss.append(trained_loss)

    plt.figure(figsize=(12, 9))
    plt.plot(candidate_dor, losses, color="red", marker="x")
    plt.plot(candidate_dor, total_loss, color="blue", marker="o")
    plt.xlabel("Dropout Rate")
    plt.ylabel("Validation/Training RMSE")
    plt.grid(True)
    plt.show()

    # want to see how the validation loss (test data) compares with total loss (train data)
    # general rule of thumb: if training loss << validation loss, increase dropout because overfitting, vice versa
    print(
        f"total losses from training set after 25 epochs {[round(l, 6) for l in total_loss]}"
    )


# def visualize():
#     # just to test out what the final thing will look like
#     x, y, scaler = preprocess_training_data("CMAPSS/train_FD001.txt")
#     x_test, y_test = preprocess_test_data(
#         "CMAPSS/test_FD001.txt", "CMAPSS/RUL_FD001.txt", scaler
#     )
#     # can replace with any train_model_XXX function, need to remove trained_loss for other train_model_XXX functions
#     trained_model, trained_loss = node.train_model_tunedor(0.1, 96, x, y)
#     node.evaluate_model_tune_visualize(x_test, y_test, trained_model)


if __name__ == "__main__":
    # general order of tuning: learning rate, hidden dimensions, dropout rate
    # important to hold seed constant as then can compare between different neural networks with variations in their parameters
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    settings: dict = {
        "batch_size": 128,
        "epochs": 10,
        "lr": 0.001,
        "hidden_dimension": 64,
        "encoder_dimension": 128,
        "regressor_dimension": 128,
        "dropout": 0.2,
    }
    dropout_rate_sweep(
        MODEL_TYPE_NODE,
        "CMAPSS/train_FD001.txt",
        settings=settings,
    )
