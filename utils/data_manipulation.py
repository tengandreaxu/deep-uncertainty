import torch
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std()


def torch_tensor_train_test_split(
    df: pd.DataFrame,
    y: pd.Series,
    test_size: Optional[float] = 0.1,
    shuffle: Optional[bool] = True,
    standardize: Optional[bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=test_size, shuffle=shuffle
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    if standardize:
        X_train = pd.DataFrame(x_scaler.fit_transform(X_train))
        X_test = pd.DataFrame(x_scaler.transform(X_test))
        y_scaler.fit(y_train.values.reshape(-1, 1))

    X_train = torch.tensor(X_train.values)
    X_test = torch.tensor(X_test.values)
    y_test = torch.tensor(y_test.values).reshape([y_test.shape[0], 1])
    y_train = torch.tensor(y_train.values).reshape([y_train.shape[0], 1])
    return X_train, X_test, y_train, y_test, x_scaler, y_scaler
