
# Churn Prediction Model

This project contains a modular Python script for building and evaluating a customer churn prediction model. The model is built using a pipeline that includes data preprocessing, various machine learning classifiers, and hyperparameter tuning with `GridSearchCV`. The best-performing model is then saved for future use.

## Project Structure

-   `Model.ipynb`: The original Jupyter Notebook where the initial exploration and model building were performed.
-   `cleaned_churn_data.csv`: The dataset used for training and testing the model.
-   `model.py`: A refactored, modular version of the notebook code.
-   `requirements.txt`: A list of all necessary libraries to run the project.
-   `LightGBM_Model.pkl`: The final trained model saved as a pickle file.

## `modular_code.py`

The `model.py` script is organized into several functions to ensure clarity and reusability:

1.  **`load_and_split_data(file_path)`**:
    -   Loads the dataset from the specified file path.
    -   Separates the features (`X`) from the target variable (`y`).
    -   Splits the data into training and testing sets using `train_test_split`.

2.  **`preprocess_data(X_train, X_test, y_train)`**:
    -   Handles the preprocessing of categorical features.
    -   Replaces 'Yes'/'No' values with `1`/`0`.
    -   Uses `ColumnTransformer` to apply different encoding techniques:
        -   `OneHotEncoder` for nominal features (`Gender`, `Internet Type`, `Payment Method`).
        -   `OrdinalEncoder` for the ordinal `Contract` feature.
        -   `TargetEncoder` for the high-cardinality `City` feature.
    -   Encodes the target variable (`Customer Status`) using `LabelEncoder`.

3.  **`train_and_evaluate_model(X_train, y_train_encoded, X_test, y_test_encoded, preprocessor)`**:
    -   Sets up a machine learning pipeline that first preprocesses the data and then applies a classifier.
    -   Defines a parameter grid to perform `GridSearchCV` on three different classifiers: `RandomForestClassifier`, `XGBClassifier`, and `LGBMClassifier`.
    -   Fits the `GridSearchCV` object to the training data to find the best model and hyperparameters.
    -   Prints the best parameters and the cross-validation score.
    -   Evaluates the best model's performance on the test set and prints the accuracy.

4.  **`save_model(model, file_name)`**:
    -   Serializes and saves the trained model to a file using `pickle`.

5.  **`load_model(file_name)`**:
    -   Loads a saved model from a pickle file.

## How to Run the Project

### Prerequisites

Make sure you have Python installed. The project was developed with Python 3.12.7.

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    ```

2.  Navigate to the project directory:
    ```bash
    cd <project-directory>
    ```

3.  Install the required libraries using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

### Execution

To run the modular script and train the model, simply execute the `src/model.py` file:

```bash
python src/model.py
