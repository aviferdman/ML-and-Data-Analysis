###### Your ID ######
# ID1: 316420132
# ID2: 313306078
#####################

# imports 
import numpy as np
import pandas as pd
import warnings

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    X = (X - np.mean(X)) / (np.max(X) - np.min(X))
    y = (y - np.mean(y)) / (np.max(y) - np.min(y))
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    # If X is a 1D array, reshape it to a 2D column vector
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    m = X.shape[0]
    bias_column = np.ones((m, 1))
    X = np.concatenate((bias_column, X), axis=1)
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """

    J = 0  # We use J for the cost.
    m = y.shape[0]

    # Use only the number of features that match theta
    X_used = X[:, :theta.shape[0]]

    predictions = np.dot(X_used, theta)
    errors = predictions - y
    J = (1 / (2 * m)) * np.sum(errors ** 2)

    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration

    m = len(y)
    for i in range(num_iters):
        # Slice X to match the shape of theta
        X_used = X[:, :theta.shape[0]]

        # Computing prediction and error
        predictions = np.dot(X_used, theta)
        errors = predictions - y

        # Computing gradient
        gradient = (1 / m) * np.dot(X_used.T, errors)
        theta = theta - alpha * gradient
        cost = compute_cost(X, y, theta)
        J_history.append(cost)
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudo inverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    pinv_theta = []
    X_transpose = X.T
    XTX = X_transpose @ X

    # Use solve instead of inv to avoid singular matrix issues
    try:
        pinv_theta = np.linalg.solve(XTX, X_transpose @ y)
    except np.linalg.LinAlgError:
        # Fallback: Add small regularization (Ridge regression trick)
        tolerance = 1e-8
        d = XTX.shape[0]
        XTX_reg = XTX + tolerance * np.eye(d)
        pinv_theta = np.linalg.solve(XTX_reg, X_transpose @ y)

    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration

    m = len(y)
    tolerance = 1e-8
    prev_cost = float('inf')  # Initial "infinite" cost

    for i in range(num_iters):
        # Use only relevant columns of X based on theta size
        X_used = X[:, :theta.shape[0]]

        predictions = np.dot(X_used, theta)
        errors = predictions - y
        gradient = (1 / m) * np.dot(X_used.T, errors)
        theta = theta - alpha * gradient

        # Computing cost and check improvement
        cost = compute_cost(X, y, theta)
        J_history.append(cost)

        if abs(prev_cost - cost) < tolerance:
            break  # Stopping here because of the cost

        prev_cost = cost

        if theta.shape[0] < X.shape[1]:
            padded_theta = np.zeros(X.shape[1])
            padded_theta[:theta.shape[0]] = theta
            theta = padded_theta

    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """

    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}  # {alpha_value: validation_loss}

    for alpha in alphas:
        np.random.seed(42)
        theta = np.random.random(size=X_train.shape[1])
        theta, J_history = efficient_gradient_descent(X_train, y_train, theta, alpha, iterations)

        alpha_dict[alpha] = compute_cost(X_val, y_val, theta)

    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []

    remaining_features = list(range(X_train.shape[1]))

    for _ in range(5):  # Select 5 features
        best_feature = None
        best_val_loss = float('inf')

        for feature in remaining_features:
            current_features = selected_features + [feature]

            # Apply bias trick
            X_train_subset = apply_bias_trick(X_train[:, current_features])
            X_val_subset = apply_bias_trick(X_val[:, current_features])

            # Initialize theta with 1s
            theta_init = np.ones(X_train_subset.shape[1])

            # Train model
            theta, _ = efficient_gradient_descent(X_train_subset, y_train, theta_init, best_alpha, iterations)

            # Compute validation loss
            val_loss = compute_cost(X_val_subset, y_val, theta)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_feature = feature

        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)

    return selected_features


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()

    for col in df.columns:
        squared_col_name = f"{col}^2"
        df_poly[squared_col_name] = df[col] ** 2

    return df_poly