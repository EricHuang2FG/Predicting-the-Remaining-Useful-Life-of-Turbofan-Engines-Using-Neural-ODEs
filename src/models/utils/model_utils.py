import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from src.models.node import ODE
from src.models.cnn_node import CNN_ODE
from src.utils.constants import (
    DATASET_ID_FD001,
    DATASET_ID_FD002,
    DATASET_ID_FD003,
    DATASET_ID_FD004,
    NUM_SETTINGS_AND_SENSOR_READINGS,
    MODEL_TYPE_CNN_NODE,
    MODEL_TYPE_NODE,
    DEFAULT_NETWORK_SETTINGS,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_FIGURE_SIZE,
    LINE_WIDTH,
)


def load_model_from_file(
    model_class: str,
    path: str = "models/ode.model",
    settings: dict = DEFAULT_NETWORK_SETTINGS,
) -> nn.Module:
    model: nn.Module = initialize_model(model_class, settings=settings)
    model.load_state_dict(torch.load(path))

    return model


def initialize_model(
    model_class: str, settings: dict = DEFAULT_NETWORK_SETTINGS
) -> nn.Module:
    if model_class not in [MODEL_TYPE_NODE, MODEL_TYPE_CNN_NODE]:
        raise ValueError(f"Unknown model_class: {model_class}")

    # for the settings dictionary, we use [] access because
    # we want the code to crash if such setting is not provided
    if model_class == MODEL_TYPE_NODE:
        return ODE(
            input_dimension=NUM_SETTINGS_AND_SENSOR_READINGS,
            hidden_dimension=settings["hidden_dimension"],
            encoder_dimension=settings["encoder_dimension"],
            regressor_dimension=settings["regressor_dimension"],
            dropout=settings["dropout"],
            sequence_length=DEFAULT_WINDOW_SIZE,
        )

    if model_class == MODEL_TYPE_CNN_NODE:
        return CNN_ODE(
            input_dimension=NUM_SETTINGS_AND_SENSOR_READINGS,
            cnn_num_kernals=settings["cnn_num_kernals"],
            cnn_kernal_size=settings["cnn_kernal_size"],
            cnn_stride=settings["cnn_stride"],
            cnn_padding=settings["cnn_padding"],
            hidden_dimension=settings["hidden_dimension"],
            encoder_dimension=settings["encoder_dimension"],
            regressor_dimension=settings["regressor_dimension"],
            dropout=settings["dropout"],
            sequence_length=DEFAULT_WINDOW_SIZE,
        )


def train_model(
    model_class: str,
    dest_path: str,
    input_data: torch.FloatTensor,
    expected_output: torch.FloatTensor,
    settings: dict = DEFAULT_NETWORK_SETTINGS,
    return_loss: bool = False,
    input_validation_data: torch.FloatTensor = None,  # pass in if early stopping is desired
    expected_validation_output: torch.FloatTensor = None,
):
    model = initialize_model(model_class, settings=settings)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])

    dataset = TensorDataset(input_data, expected_output)
    dataloader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True)
    model.train()

    lowest_validation_loss = float("inf")
    num_no_improvement_epochs = 0
    optimal_state = None

    epochs = settings["epochs"]
    for epoch in range(epochs):
        total_loss_per_epoch = 0
        for x, y in dataloader:
            optimizer.zero_grad()  # zero all the gradients first
            predictions = model(x)  # CALLS MODEL TO GET THE MODEL'S PREDICTION
            loss = loss_function(predictions, y)
            loss.backward()  # BACK PROPAGATION, gradient descent of the (take gradient of) loss function
            # chain rule only applies in backward pass, but infinite layers
            # so instead of multiplying the derivates (in chain rule) integrate them
            # known as the adjoint method
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  # updates the parameters/weights taking into account those gradient values, and size of each step is determined by learning rate
            total_loss_per_epoch += loss.item()

        print(
            f"Epoch [{epoch + 1}/{epochs}], loss: {(total_loss_per_epoch / len(dataloader)):.6f}"
        )

        # check using validation set to see if early stopping is required
        if input_validation_data is not None and expected_validation_output is not None:
            model.eval()
            with torch.no_grad():
                validation_predictions = model(input_validation_data)
                validation_loss = loss_function(
                    validation_predictions, expected_validation_output
                ).item()
                print(f"Current validation loss: {validation_loss}")

            if (
                validation_loss <= lowest_validation_loss - 1e-4
            ):  # if it's better, to a 4 decimal places tolerance
                lowest_validation_loss = validation_loss
                num_no_improvement_epochs = 0
                optimal_state = model.state_dict()
            else:
                num_no_improvement_epochs += 1

            if num_no_improvement_epochs >= 5:
                print(
                    f"Early stopping at [{epoch + 1}/{epochs}], validation loss: {(lowest_validation_loss):.6f}"
                )
                model.load_state_dict(optimal_state)
                break

    if optimal_state is not None:
        model.load_state_dict(optimal_state)

    torch.save(model.state_dict(), dest_path)

    if return_loss:
        return model, total_loss_per_epoch / len(
            dataloader
        )  # training loss at last epoch
    return model


def evaluate_model(
    input_data: torch.FloatTensor,
    expected_output: torch.FloatTensor,
    model_class: str,
    model: nn.Module,  # pass this in directly, or, pass in None here but give the model_path
    model_path: str = "models/ode.model",
    settings: dict = DEFAULT_NETWORK_SETTINGS,
    plot: bool = True,
    figure_dest: str = "figures/untitled_figure.pdf",
) -> tuple[float, float]:
    if not model:
        model = load_model_from_file(model_class, path=model_path, settings=settings)
    model.eval()

    with torch.no_grad():
        predictions: torch.FloatTensor = model(input_data)

    prediction_array: np.ndarray = predictions.numpy()
    ground_truth_array: np.ndarray = expected_output.numpy()

    # clip to be below 100 (the paper does this too)
    # prediction_array = np.minimum(100, prediction_array)
    ground_truth_array = np.minimum(100, ground_truth_array)

    # clip the value in the array to be meaningful values between 0 and 100
    # similarly to the RUL segmentation done in the reference paper
    # for obviously bad values like -1500 (there are about 4 of them out of 100)
    # we replace with a default of 100
    prediction_array_cleaned: list = [
        value if 0 <= value <= 100 else 100 for value in prediction_array
    ]
    prediction_array_cleaned: np.ndarray = np.array(prediction_array_cleaned)
    ground_truth_array = np.minimum(100, ground_truth_array)

    sort_indicies: np.ndarray = np.argsort(ground_truth_array)
    ground_truth_sorted: np.ndarray = ground_truth_array[sort_indicies]
    prediction_array_cleaned_sorted: np.ndarray = prediction_array_cleaned[
        sort_indicies
    ]

    rmse = np.sqrt(np.mean((prediction_array_cleaned - ground_truth_array) ** 2))

    non_zero_mask = ground_truth_array != 0
    mape = np.mean(
        np.abs(
            (
                prediction_array_cleaned[non_zero_mask]
                - ground_truth_array[non_zero_mask]
            )
            / ground_truth_array[non_zero_mask]
        )
    )

    print(f"RMSE: {rmse}, MAPE: {mape}")

    if plot:
        # extract the dataset id by inferring from the model names
        # if such extraction fails, the dataset id will not be included in the graph title
        dataset_id = ""
        if model_path:
            if "001" in model_path:
                dataset_id = DATASET_ID_FD001
            elif "002" in model_path:
                dataset_id = DATASET_ID_FD002
            elif "003" in model_path:
                dataset_id = DATASET_ID_FD003
            elif "004" in model_path:
                dataset_id = DATASET_ID_FD004

        # plotting
        x = list(range(len(prediction_array)))

        plt.figure(figsize=DEFAULT_FIGURE_SIZE)
        plt.plot(
            x,
            ground_truth_sorted,
            color="red",
            label="Actual RUL",
            linewidth=LINE_WIDTH,
        )
        plt.plot(
            x,
            prediction_array_cleaned_sorted,
            color="purple",
            label=f"Predicted RUL, RMSE: {rmse:.4f}, MAPE: {mape:.4f}",
            linewidth=LINE_WIDTH,
        )

        plt.xlabel("Engine Number", fontweight="bold", fontsize=22)
        plt.ylabel("RUL", fontweight="bold", fontsize=22)
        plt.title(
            "Actual and Predicted RUL" + ("" if not dataset_id else f", {dataset_id}"),
            fontweight="bold",
            fontsize=26,
        )
        plt.legend(loc="lower right", fontsize=18)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.grid(True)
        plt.savefig(figure_dest, dpi=300)
        plt.tick_params(axis="both", which="major", labelsize=15)
        plt.tight_layout()

        plt.show()

    return rmse, mape


# def train_model_tunelr(lr, input_data, expected_output, epochs=5):
#     model = ODE(input_dimension=NUM_SETTINGS_AND_SENSOR_READINGS, hidden_dimension=64)
#     loss_function = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr)
#     dataset = TensorDataset(input_data, expected_output)
#     dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for x, y in dataloader:
#             optimizer.zero_grad()
#             predictions = model(x)
#             loss = loss_function(predictions, y)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             total_loss += loss.item()

#         print(
#             f"Epoch [{epoch + 1}/{epochs}], loss: {(total_loss / len(dataloader)):.6f}"
#         )
#     # return model directly
#     return model


# def train_model_tunehd(hd, input_data, expected_output, epochs=5):
#     model = ODE(
#         input_dimension=NUM_SETTINGS_AND_SENSOR_READINGS, hidden_dimension=hd
#     )  # changed hidden dimension here
#     loss_function = nn.MSELoss()
#     optimizer = torch.optim.Adam(
#         model.parameters(), lr=0.001
#     )  # changed the lr to be optimized

#     dataset = TensorDataset(input_data, expected_output)
#     dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for x, y in dataloader:
#             optimizer.zero_grad()
#             predictions = model(x)
#             loss = loss_function(predictions, y)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             total_loss += loss.item()

#         print(
#             f"Epoch [{epoch + 1}/{epochs}], loss: {(total_loss / len(dataloader)):.6f}"
#         )
#     return model


# def train_model_tunedor(dor, hd, input_data, expected_output, epochs=15):
#     model = ODE(
#         input_dimension=NUM_SETTINGS_AND_SENSOR_READINGS,
#         hidden_dimension=hd,
#         dropout=dor,
#     )
#     loss_function = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     dataset = TensorDataset(input_data, expected_output)
#     dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for x, y in dataloader:
#             optimizer.zero_grad()
#             predictions = model(x)
#             loss = loss_function(predictions, y)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             total_loss += loss.item()

#         print(
#             f"Epoch [{epoch + 1}/{epochs}], loss: {(total_loss / len(dataloader)):.6f}"
#         )

#     avg_loss = total_loss / len(dataloader)
#     return (model, avg_loss)
#     # returns tuple that includes the avg loss from training at the final 25th epoch


# def evaluate_model_tunelr(
#     input_data: torch.FloatTensor,
#     expected_output: torch.FloatTensor,
#     model: torch.nn.Module,
#     input_dimension: int = NUM_SETTINGS_AND_SENSOR_READINGS,
#     hidden_dimension: int = 64,
#     sequence_length: int = 40,
# ) -> float:
#     model.eval()

#     with torch.no_grad():
#         predictions: torch.FloatTensor = model(input_data)

#     prediction_array: np.ndarray = predictions.numpy()
#     ground_truth_array: np.ndarray = expected_output.numpy()

#     prediction_array = np.minimum(100, prediction_array)
#     ground_truth_array = np.minimum(100, ground_truth_array)

#     prediction_array_cleaned: list = [
#         value if 0 <= value <= 100 else 100 for value in prediction_array
#     ]
#     prediction_array_cleaned: np.ndarray = np.array(prediction_array_cleaned)

#     rmse = np.sqrt(np.mean((prediction_array_cleaned - ground_truth_array) ** 2))
#     return rmse


# def evaluate_model_tune_visualize(
#     input_data: torch.FloatTensor,
#     expected_output: torch.FloatTensor,
#     model: torch.nn.Module,
# ) -> float:
#     model.eval()

#     with torch.no_grad():
#         predictions: torch.FloatTensor = model(input_data)

#     prediction_array: np.ndarray = predictions.numpy()
#     ground_truth_array: np.ndarray = expected_output.numpy()

#     prediction_array = np.minimum(100, prediction_array)
#     ground_truth_array = np.minimum(100, ground_truth_array)

#     prediction_array_cleaned: list = [
#         value if 0 <= value <= 100 else 100 for value in prediction_array
#     ]
#     prediction_array_cleaned: np.ndarray = np.array(prediction_array_cleaned)

#     sort_indicies: np.ndarray = np.argsort(ground_truth_array)
#     ground_truth_sorted: np.ndarray = ground_truth_array[sort_indicies]
#     prediction_array_cleaned_sorted: np.ndarray = prediction_array_cleaned[
#         sort_indicies
#     ]

#     x = list(range(len(prediction_array)))

#     plt.plot(x, ground_truth_sorted, color="red")
#     plt.plot(x, prediction_array_cleaned_sorted, color="purple")
#     plt.show()

#     rmse = np.sqrt(np.mean((prediction_array_cleaned - ground_truth_array) ** 2))
#     print(rmse)
