def sqrt_loss(y_hat, y):
    result_vec = (y_hat - y) ** 2 / 2
    return result_vec.mean()