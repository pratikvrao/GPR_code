from sklearn.preprocessing import StandardScaler


def scale_data(X_train, X_test, y_train, y_test):
    """ Scale the data (X,y) to have the same order of magnitude """

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train_scaled = scaler_X.transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y


def inverse_transform(scaler, *arrays):
    """Inverse transform and reshape arrays to 2D if needed."""
    return [scaler.inverse_transform(arr.reshape(-1, 1)) if arr.ndim == 1 else scaler.inverse_transform(arr)
            for arr in arrays]
