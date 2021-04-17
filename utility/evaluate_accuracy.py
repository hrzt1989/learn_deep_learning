def evaluate_accuracy(data_iter, net, w, b ):
    acc_sum = 0
    n = 0
    for x, y in data_iter:
        y_hat = net(x, w, b)
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def evaluate_accuracy_v1(data_iter, net,  hidden_ws, hidden_bs, output_w, output_b):
    acc_sum = 0
    n = 0
    for x, y in data_iter:
        y_hat = net(x, hidden_ws, hidden_bs, output_w, output_b)
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def evaluate_accuracy_v2(data_iter, net):
    acc_sum = 0
    n = 0
    for x, y in data_iter:
        y_hat = net(x)
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n