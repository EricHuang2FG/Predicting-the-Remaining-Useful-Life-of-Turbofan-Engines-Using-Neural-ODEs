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
    """initialize a PyTorch model and load the weights stored in a file at path

    The PyTorch model is first initialized with passed-in settings using initialize_model(),
    then, it is loaded with weights stored as a file at path

    Args:
        model_class (str): class of the model. Either MODEL_TYPE_NODE or MODEL_TYPE_CNN_NODE.
        path (str): file path of the file storing the weights.
        settings (dict): a dictionary of settings of the particular model belonging to model_class.

    Returns:
        nn.Module: model loaded with weights stored in the file at path.
    """

    # load the saved weights from file path into a new instance of a model
    model: nn.Module = initialize_model(model_class, settings=settings)
    model.load_state_dict(torch.load(path))

    return model


def initialize_model(
    model_class: str, settings: dict = DEFAULT_NETWORK_SETTINGS
) -> nn.Module:
    """initialize a PyTorch model with passed-in settings

    An empty PyTorch model (i.e. model with non-determined weights and biases) of model_class with
    passed-in settings is initialized and returned

    Args:
        model_class (str): class of the model. Either MODEL_TYPE_NODE or MODEL_TYPE_CNN_NODE.
        settings (dict): a dictionary of settings of the particular model belonging to model_class.

    Returns:
        nn.Module: initialized model of model_class.
    """

    if model_class not in [MODEL_TYPE_NODE, MODEL_TYPE_CNN_NODE]:
        raise ValueError(f"Unknown model_class: {model_class}")
    # for the settings dictionary, we use [] access because
    # we want the code to crash if such setting is not provided

    # create instance of model with basic settings
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
    """train a PyTorch model of model_class

    Train a PyTorch model of model_class with passed-in settings. The weights of the trained model is
    saved at dest_path. If input_validation_data and expected_validation_output are not None,
    then early stopping is employed.

    Args:
        model_class (str): class of the model. Either MODEL_TYPE_NODE or MODEL_TYPE_CNN_NODE.
        dest_path (str): destination location where the trained weights are to be saved
        input_data (torch.FloatTensor): data for training
        expected_output (torch.FloatTensor): labels for training data
        settings (dict): a dictionary of settings of the particular model belonging to model_class.
        return_loss (bool): whether or not validation loss should be returned by the function.
        input_validation_data (torch.FloatTensor): validation data.
        expected_validation_output (torch.FloatTensor): labels for validation data

    Returns:
        nn.Module: trained model. If return_loss is true, the average training loss for the last epoch is returned in a tuple
    """
    model = initialize_model(model_class, settings=settings)

    # Mean Squared Error Loss, difference between prediction and true values
    loss_function = nn.MSELoss()
    # use Adam Algorithm to update weights to minimize loss
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])

    dataset = TensorDataset(input_data, expected_output)
    dataloader = DataLoader(
        dataset, batch_size=settings["batch_size"], shuffle=True
    )  # feeds batches of data to train,
    # data is randomly ordered at beginning of each epoch
    model.train()

    lowest_validation_loss = float("inf")
    num_no_improvement_epochs = 0
    optimal_state = None

    # going through epochs one at a time
    epochs = settings["epochs"]
    for epoch in range(epochs):
        total_loss_per_epoch = 0
        # going through each pass of the training set (epoch) in batches
        for x, y in dataloader:
            optimizer.zero_grad()  # zero all the gradients first to reset model weights, only want the error associated with current batch
            predictions = model(x)
            # calls model to get the mode's prediction for the given input values
            # goes through a forward pass (automatically calls forward fucntion of the model type)
            loss = loss_function(predictions, y)
            loss.backward()
            # back propagation: figures out how much each parameter contributes to the final loss, take derivatives as look at changes
            # calculates gradient of the loss function wrt each parameter
            # hidden layers creates composite functions, thus require chain rule to find impact of earlier parameters on loss
            # chain rule takes into account gradient calculated at a later layer to find the gradient for an earlier layer

            # for Neural ODEs, this is too complicated due to its infinite dimensions
            # thus, use Adjoint Method that involves integrating another ODE backwards in time to find the assocaiated gradients with each parameter
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  # updates the parameters/weights taking into account gradient values
            # size of each step is determined by learning rate
            total_loss_per_epoch += loss.item()

        print(
            f"Epoch [{epoch + 1}/{epochs}], loss: {(total_loss_per_epoch / len(dataloader)):.6f}"
        )

        # check using validation set to see if early stopping is required
        if input_validation_data is not None and expected_validation_output is not None:
            model.eval()
            with torch.no_grad():  # autograd disabled
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
                optimal_state = (
                    model.state_dict()
                )  # saves a copy of best parameters at this point
            else:
                num_no_improvement_epochs += 1

            if num_no_improvement_epochs >= 5:
                print(
                    f"Early stopping at [{epoch + 1}/{epochs}], validation loss: {(lowest_validation_loss):.6f}"
                )
                model.load_state_dict(
                    optimal_state
                )  # restores the best model parameters
                break

    if optimal_state is not None:
        model.load_state_dict(optimal_state)

    torch.save(model.state_dict(), dest_path)

    if return_loss:
        return model, total_loss_per_epoch / len(
            dataloader
        )  # average training loss per batch for last epoch
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
    """evaluates a PyTorch model

    Evaluates a PyTorch model of model_class by computing the RMSE and MAPE of the predicted RUL with the
    expected_output. The model is either directly passed in as an argument, or loaded as a file.
    If plot is True, graphs that visualizes predicted RULs and the corresponding expected_output will be plotted,
    and saved at figure_dest

    Args:
        input_data (torch.FloatTensor): data for evaluation.
        expected_output (torch.FloatTensor): ground-truth RULs for evaluation.
        model_class (str): class of the model. Either MODEL_TYPE_NODE or MODEL_TYPE_CNN_NODE.
        model (nn.Module): PyTorch model of the model to be evaluated.
        model_path (str): file path containing the weights of the model to be evaluated.
        settings (dict): a dictionary of settings of the particular model belonging to model_class.
        plot (bool): whether or not graphs visualizing the predicted and actual RULs should be produced
        figure_dest (str): if plot is True, the destination location where the plotted graph should be saved

    Returns:
        tuple[float, float]: a tuple of the RMSE and MAPE values of the model when evaluated on input_data
    """
    if not model:
        model = load_model_from_file(model_class, path=model_path, settings=settings)
    model.eval()

    with torch.no_grad():
        predictions: torch.FloatTensor = model(input_data)

    prediction_array: np.ndarray = predictions.numpy()
    ground_truth_array: np.ndarray = expected_output.numpy()

    # clip to be below 100
    # prediction_array = np.minimum(100, prediction_array)
    ground_truth_array = np.minimum(100, ground_truth_array)

    # clip the value in the array to be meaningful values between 0 and 100
    # for bad values like -1500 that do not correspond with physical meaning (~4 of them out of 100)
    # replace with a default of 100
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

    # avoid division by 0
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
        plt.scatter(
            x,
            prediction_array_cleaned_sorted,
            color="purple",
            label=f"Predicted RUL, RMSE: {rmse:.4f}, MAPE: {mape:.4f}",
            s=16,
            # linewidth=LINE_WIDTH,
        )

        plt.xlabel("Engines", fontweight="bold", fontsize=22)
        plt.ylabel("RUL", fontweight="bold", fontsize=22)
        plt.title(
            f"Actual and Predicted RUL of {"CNN-NODE" if model_class == "cnn_node" else "NODE"}"
            + ("" if not dataset_id else f", {dataset_id}"),
            fontweight="bold",
            fontsize=26,
        )
        plt.legend(loc="lower right", fontsize=18)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.grid(True)
        plt.savefig(figure_dest, dpi=300)
        plt.tick_params(axis="both", which="major", labelsize=22)
        plt.tight_layout()

        plt.show()

    return rmse, mape
