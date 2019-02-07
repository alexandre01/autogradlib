def LossMSE(output, target):
    N = output.tensor.size(0)
    return (output - target).pow(2).sum() / (2 * N)