from src.models.utils.model_utils import evaluate_model
from src.utils.data_processing import preprocess_training_data, preprocess_test_data
from src.utils.constants import MODEL_TYPE_NODE


def main() -> None:
    settings: dict = {
        "batch_size": 128,
        "epochs": 25,
        "lr": 0.001,
        "hidden_dimension": 64,
        "encoder_dimension": 128,
        "regressor_dimension": 32,
        "dropout": 0.2,
    }

    x, y, scaler = preprocess_training_data("CMAPSS/train_FD001.txt")
    # before training of model
    # model = node.train_model("models/ode.FD001.v3.model", x, y)

    x_test, y_test = preprocess_test_data(
        "CMAPSS/test_FD001.txt", "CMAPSS/RUL_FD001.txt", scaler
    )
    evaluate_model(
        x_test,
        y_test,
        MODEL_TYPE_NODE,
        None,
        "models/ode.FD001.v3.model",
        settings=settings,
    )


if __name__ == "__main__":
    main()
