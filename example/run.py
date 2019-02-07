from torch import FloatTensor
import math

from autogradlib import Variable
from autogradlib.module import Sequential, Linear, ReLU, Softmax
from autogradlib.Functional import LossMSE


def generate_data(N):
    input = FloatTensor(N, 2).uniform_()
    target = (((input - 0.5).pow(2).sum(1)) <= 1 / (2 * math.pi)).long().view(-1, 1)
    target_one_hot = FloatTensor(N, 2).zero_()
    target_one_hot.scatter_(1, target, 1)

    return input, target_one_hot


def compute_metrics(model, input, target):
    output = model(input)
    loss = LossMSE(output, target).tensor[0]
    accuracy = (output.tensor.max(dim=1)[1] == target.tensor.max(dim=1)[1]).sum() / target.tensor.size(0)
    return loss, accuracy


if __name__ == "__main__":
    # Defining the parameters
    N, eta, nb_epochs = 1000, 1e-1, 500

    # Generating input and target data for training and testing
    train_input, train_target = generate_data(N)
    validation_input, validation_target = generate_data(N)
    test_input, test_target = generate_data(N)

    mean, std = train_input.mean(), train_input.std()

    train_input.sub_(mean).div_(std)
    validation_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    train_input, train_target = Variable(train_input, name="x"), Variable(train_target, name="target")
    validation_input, validation_target = Variable(validation_input), Variable(validation_target)
    test_input, test_target = Variable(test_input), Variable(test_target)

    # Defining the model
    model = Sequential(
        Linear(2, 25), ReLU(),
        Linear(25, 25), ReLU(),
        Linear(25, 25), ReLU(),
        Linear(25, 2), Softmax(dim=1)
    )
    print("Model:", model)

    # Drawing the tree of gradient operations
    l = LossMSE(model(train_input), train_target)
    l.name = "MSE Loss"
    dot = l.draw_graph()
    dot.render("plots/model.gv", view=True)

    # Training
    print("Training...")
    for e in range(nb_epochs):
        loss = LossMSE(model(train_input), train_target)
        print("Epoch {}, loss: {}".format(e + 1, loss.tensor[0]), end="\r")

        train_loss, train_accuracy = compute_metrics(model, train_input, train_target)
        validation_loss, validation_accuracy = compute_metrics(model, validation_input, validation_target)

        # Backprop
        model.zero_grad()
        loss.backward()

        for param in model.params():
            param.tensor -= eta * param.grad

    print("\nDone.")

    # Testing
    train_loss, train_accuracy = compute_metrics(model, train_input, train_target)
    print("Train loss: {}, train accuracy: {}".format(train_loss, train_accuracy))

    test_loss, test_accuracy = compute_metrics(model, test_input, test_target)
    print("Test loss: {}, test accuracy: {}".format(test_loss, test_accuracy))
