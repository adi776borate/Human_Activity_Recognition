from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    correct_pred = (y_hat == y).sum()
    accuracy = correct_pred/y.size
    return accuracy


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    true_positive = ((y_hat == cls) & (y == cls)).sum()
    predicted_positive = (y_hat == cls).sum()
    if predicted_positive == 0:
        return 0.0
    
    precision = true_positive / predicted_positive
    return precision


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    true_positive = ((y_hat == cls) & (y == cls)).sum()
    actual_positive = (y == cls).sum()
    if actual_positive == 0:
        return 0.0
    
    recall = true_positive / actual_positive
    return recall


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    residuals = y_hat - y
    mse = (residuals ** 2).mean()
    rmse = np.sqrt(mse)
    return rmse
    


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    residuals = y_hat - y
    mae = residuals.abs().mean()
    return mae

