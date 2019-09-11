import torch
import torch.nn.functional as fn
import pdb

class SkipGram:
    # Network consists of an input layer, a single hidden layer and an output layer to which
    # we apply the softmax function. There are two weight matrices which contain the embedding
    # information, the first between the input and hidden layers, and the second between the
    # hidden and output layers.

    # Class constructor. v_size is the size of the vocabulary, also the length of the one-hot-
    # encoding input vector. d_size is the size of the hidden layer vector.
    def __init__(self, v_size, d_size):
        # Store vocabulary and hidden layer sizes for use later.
        self.vocab_size = v_size
        self.hidden_size = d_size

        # Randomly initialise weight matrices with required dimensions.
        self.weight_1 = torch.randn(v_size, d_size)
        self.weight_2 = torch.randn(d_size, v_size)

    # Function feeds input word vector (x) through model to produce a vector containing a
    # prediction of the words within a context window of the input word.
    def feedforward(self, x):
        # Multiply the input vector by the transpose of the first weight matrix to get
        # the hidden layer vector.
        h = torch.matmul(self.weight_1.transpose(0, 1), x)

        # Multiply the hidden layer by the transpose of the second weight matrix to get
        # the output vector.
        u = torch.matmul(self.weight_2.transpose(0, 1), h)

        # Apply the softmax function to the output vector.
        y = fn.softmax(u)

        # Return the resultant vector.
        return y

    # Function performs backpropagation of negative log loss between expected and actual
    # values through the weights of the network. Returns the derivative of the loss
    # function with respect the two weight matrices.
    def backprop(self, x, y_act):
        # Feed input word vector through network to generate a prediction, and store the
        # relevant intermediary vectors for use in backpropagation.
        h = torch.matmul(self.weight_1.transpose(0, 1), x)
        u = torch.matmul(self.weight_2.transpose(0, 1), h)
        y_pred = fn.softmax(u)

        # Derivative of negative log loss (L) with respect to the second weight matrix.
        # Equates to the outer product between the hidden vector (h) and the difference
        # between the predicted and actual output vectors.
        dw2 = h.ger(y_pred - y_act)

        # Derivative of negative log loss (L) with respect to the first weight matrix.
        # Equates to the outer product between the input vector (x) and the difference
        # between the predicted and actual output vectors, multiplied by the second
        # weight matrix.
        dw1 = x.ger(torch.matmul(self.weight_2, (y_pred - y_act)))
        
        # Return the loss function derivatives with respect to the two weight matrices.
        return dw1, dw2

    # Function trains model on provided training data, consisting of pairs of input and
    # output word vectors. The output word vector should contain all the words within
    # the context window of the input word. Training will be conducted for the specified
    # number of epochs at the specified learning rate. There is also the option to
    # provide validation data to allow training progress to be monitored.
    def train(self, training_data, epochs, lr, validation_data=None):
        
        for i in range(1, epochs):
            
            for x, y in training_data:

                dw1, dw2 = self.backprop(x, y)

                self.weight_1 = self.weight_1 - lr*dw1
                self.weight_2 = self.weight_2 - lr*dw2


            if validation_data:
                test_results = [(self.feedforward(x), y) for (x, y) in validation_data]
                num_correct = sum(int(torch.equal(y_pred, y_act))
                        for (y_pred, y_act) in test_results)
                print("Epoch {0}: {1} / {2}".format(i, num_correct, len(test_results)))


sg = SkipGram(10, 5)
pdb.set_trace()
