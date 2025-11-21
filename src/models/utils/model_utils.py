import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from src.models.node import ODE
from src.utils.constants import (
    NUM_SETTINGS_AND_SENSOR_READINGS,
    MODEL_TYPE_CNN_NODE,
    MODEL_TYPE_NODE,
    DEFAULT_NETWORK_SETTINGS,
    DEFAULT_WINDOW_SIZE,
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
    # default value same for now, but will change later
    return ODE(
        input_dimension=NUM_SETTINGS_AND_SENSOR_READINGS,
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
):
    model = initialize_model(model_class, settings=settings)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])

    dataset = TensorDataset(input_data, expected_output)
    dataloader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True)
    model.train()

    epochs = settings["epochs"]
    for epoch in range(epochs):
        total_loss = 0
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
            total_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{epochs}], loss: {(total_loss / len(dataloader)):.6f}"
        )

    torch.save(model.state_dict(), dest_path)

    if return_loss:
        return model, loss
    return model


def evaluate_model(
    input_data: torch.FloatTensor,
    expected_output: torch.FloatTensor,
    model_class: str,
    model: nn.Module,  # pass this in directly, or, pass in None here but give the model_path
    model_path: str = "models/ode.model",
    settings: dict = DEFAULT_NETWORK_SETTINGS,
    plot: bool = True,
) -> tuple[float, float]:
    if not model:
        model = load_model_from_file(model_class, path=model_path, settings=settings)
    model.eval()

    with torch.no_grad():
        predictions: torch.FloatTensor = model(input_data)

    prediction_array: np.ndarray = predictions.numpy()
    ground_truth_array: np.ndarray = expected_output.numpy()

    # clip to be below 100 (the paper does this too)
    prediction_array = np.minimum(100, prediction_array)
    ground_truth_array = np.minimum(100, ground_truth_array)

    # clean out bad values like -1500 (there are about 4 of them out of 100)
    prediction_array_cleaned: list = [
        value if 0 <= value <= 100 else 100 for value in prediction_array
    ]
    prediction_array_cleaned: np.ndarray = np.array(prediction_array_cleaned)

    sort_indicies: np.ndarray = np.argsort(ground_truth_array)
    ground_truth_sorted: np.ndarray = ground_truth_array[sort_indicies]
    prediction_array_cleaned_sorted: np.ndarray = prediction_array_cleaned[
        sort_indicies
    ]

    if plot:
        x = list(range(len(prediction_array)))

        plt.plot(x, ground_truth_sorted, color="red")
        plt.plot(x, prediction_array_cleaned_sorted, color="purple")
        plt.show()

    # sum_squared_error: int = 0
    # for index, value in enumerate(prediction_array):
    #     clipped_value = value if abs(value) <= 100 else 100
    #     sum_squared_error += (clipped_value - ground_truth_array[index]) ** 2
    # root_mean_squared_error: float = (
    #     sum_squared_error / prediction_array.shape[0]
    # ) ** 0.5
    # print(root_mean_squared_error)

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

    print(rmse)
    print(mape)

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
