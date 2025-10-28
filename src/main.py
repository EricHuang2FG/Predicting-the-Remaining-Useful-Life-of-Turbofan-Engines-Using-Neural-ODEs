import src.node as node
from src.data_processing import preprocess_training_data, preprocess_test_data


def main() -> None:
    x, y, scaler = preprocess_training_data("CMAPSS/train_FD001.txt")

    # right now we just use an epoch of 25
    # model = node.train_model("models/ode.FD001.v3.model", x, y)

    x_test, y_test = preprocess_test_data(
        "CMAPSS/test_FD001.txt", "CMAPSS/RUL_FD001.txt", scaler
    )
    node.evaluate_model(x_test, y_test, "models/ode.FD001.v3.model")


if __name__ == "__main__":
    main()
