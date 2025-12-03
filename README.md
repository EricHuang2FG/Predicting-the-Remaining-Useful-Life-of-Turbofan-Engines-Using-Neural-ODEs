# Predicting the Remaining Useful Life of Turbofan Engines Using CNN-Neural ODEs

The aim of this project is to determine the efficacy of using neural networks to predict the Remaining Useful Life (RUL) of jet turbofan engines using the [CMAPSS Jet Engine Simulated Dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data). This is an important field of research as the consequences of unexpected jet engine failure is so devastating that it is important for engineers to know remaining service life estimates in order to enact proper preventative measures. Although there currently exists deep learning approaches–primarily LSTMs–to predict RUL, the motivation behind the use of a Neural ODE (NDOE) stems from the fact that the engine sensor readings form a time continuous dataset. As such, we hypothesize that the underlying changes in sensor data can be modeled by an ODE of the form $\frac{d \mathbf{h}(t)}{dt} = f(\mathbf{h}(t), t, \theta)$ and a neural network can model the function $f(\mathbf{h}(t), t, \theta)$ which govern the instantaneous derivatives.

The project is inspired off of the [Prediction of remaining Useful  Life of Aero-engines Based on CNN-LSTM-Attention](https://link.springer.com/article/10.1007/s44196-024-00639-w) paper. Thus, the results of the Neural ODE is benchmarked against the results in this paper to see if there is an improvement in prediction accuracy as a result of using a different underlying neural network structure. The criteria used for comaprison was the Root Measure Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE). We were also interested in how feeding the data through a CNN prior to the ODE-to preemptively isolate patterns in the data-would affect the RUL predictions.

#### src/main.py - Outputting the Graphs and Final Results:
Only the main.py file needs to be compiled and run to generate the 8 graphs--4 engine data sets (FD001, FD002, FD003, FD004) run twice for both CNN_NODE and NODE models--and their associated RMSE and MAPE values. The models weights are stored under /models. The main.py file loads the pre-trained model weigths into the specific type of model (CNN_NODE or NODE), evaluates RMSE and MAPE, and plots a graph comparing the predicted and ground truth RUL values.

### CMAPSS Jet Engine Simulated Data
The CMAPSS dataset is stored udner /CMAPSS. Each FD00# file corresponds to the particular # tested jet turbofan engine in simulation. Included are the training data to train the models with the sensor readings and its assocaited measured RUL and testing data with ground truth measured RUL to validate the trained model. Each engine has many sensors with different readings. For files of the type _______:
Data is of size (num_lines, 26)
Column 1: condition number
Column 2: time measured in number of cycles
Column 3 to 5: operational settings
Column 6 to end: sensor measurements

Columns 3-26 were used for training the models.

### Setting Up Virtual Environment and Installing the Required Dependencies
1. In the command terminal, create a virtual environment through the following command: `python3 -m venv .venv`
2. Activate the virtual environment: `.venv\Scripts\activate` (Windows)
                                     `source .venv/bin/activate` (macOS/Linux)
3. Install the required libraries from requirements.txt: `pip install -r requirements.txt`
4. Select the python interpreter to be the one inside .venv: `ctrl + ship + p`, choose `Select Interpreter` and choose the recommended option `.\.venv\Scripts\python.exe`


## Summary of Each File's Purpose:

### src/utils/data_processing.py
The data processing for the CMAPSS dataset closely follows the method followed in the aforementioned paper. EXPLAIN MORE

### src/utils/constants.py
This file contains all the constants used thorughout all the files. Examples include two dictionaries `OPTIMIZED_CNN_NODE_SETTINGS` and `OPTIMIZED_NODE_SETTINGS`, one corresponding to CNN_NODE models and the other to NODE models respectively, that describe the values of the hyperparameters to be used for the model. It also specifies the kernel size.

### src/models/node.py


### src/models/cnn_node.py




### src/models/utils/model_utils.py


### src/models/utils/tune_models.py
In this file, models are trained based on the first 70% of the training data for each engine and is then evaluated on the rest of the 30% of the training data. It contains functions `learning_rate_sweep`, `hidden_dimensions_sweep`, and `dropout_rate_sweep` that sweep through different values of those respective hyperparameters to find the best value for each hyperparameter that results in the lowest RMSE values. For example, in choosing the best learning rate, the `lr` entry of the settings/hyperparameter dictionary for the model is swapped with all five candidate learning rate values in the array in a for loop: `candidate_lrs = [0.1, 0.01, 0.005, 0.001, 0.0003]`

For the hidden dimensions, if the specific model being trained is CNN_NODE, the array is `[2, 3, 4]` oterwise if it is NODE, the array is `[32, 64, 128]`.

Furthermore, it is important to note that the seed is held constant throughout the process of training the models with different values of hyperparameters to make the results reproducibl. Only then can differences noted in RMSE values be attributed to changes in hyperparameter values and not due to favourable randomly initialized values.




***Predicting the Remaining Useful Life of Turbofan Engines Using Neural ODEs***

Please do not push directly to the main branch!!!!