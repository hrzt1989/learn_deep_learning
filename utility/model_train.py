from torch import optim
from utility.evaluate_accuracy import evaluate_accuracy_v2, evaluate_accuracy_v1
from utility.sgd import sgd

def train_v0(train_data_iter,
             test_data_iter,
             net,
             hidden_ws,
             hidden_bs,
             output_w,
             output_b,
             loss,
             batch_size,
             num_epochs,
             learn_rate,):

    params = []
    for one_w in hidden_ws:
        params.append(one_w)

    for one_b in hidden_bs:
        params.append(one_b)
    params.append(output_w)
    params.append(output_b)
    for epoch in range(num_epochs):
        for x, y in train_data_iter:
            y_hat = net(x, hidden_ws, hidden_bs, output_w, output_b, is_training = True)
            loss_result = loss(y_hat, y)
            loss_sum = loss_result.sum()
            loss_sum.backward()
            sgd(params, learn_rate, batch_size)
            for one_param in params:
                one_param.grad.data.zero_()
        train_acc = evaluate_accuracy_v1(train_data_iter, net, hidden_ws, hidden_bs, output_w, output_b)
        test_acc = evaluate_accuracy_v1(test_data_iter, net, hidden_ws, hidden_bs, output_w, output_b)
        print('epoch', epoch, 'train_acc', train_acc, 'test_acc', test_acc)

def train(train_data_iter,
          test_data_iter,
          net,
          loss,
          num_epochs,
          learn_rate,
          ):
    optimizer = optim.SGD(net.parameters(), learn_rate)
    for epoch in range(num_epochs):
        for x, y in train_data_iter:
            y_hat = net(x)
            loss_sum = loss(y_hat, y).sum()
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
        net.eval()
        train_acc = evaluate_accuracy_v2(train_data_iter, net)
        test_acc = evaluate_accuracy_v2(test_data_iter, net)
        net.train()
        print('epoch', epoch, 'train_acc', train_acc, 'test_acc', test_acc)