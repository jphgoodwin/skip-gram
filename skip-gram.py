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
        
        # pdb.set_trace()
        # Return the loss function derivatives with respect to the two weight matrices.
        return dw1, dw2

    # Function trains model on provided training data, consisting of pairs of input and
    # output word vectors. The output word vector should contain all the words within
    # the context window of the input word. Training will be conducted for the specified
    # number of epochs at the specified learning rate. There is also the option to
    # provide validation data to allow training progress to be monitored.
    def train(self, training_data, epochs, lr, validation_data=None):
        # Train for the specified number of epochs.
        for i in range(1, epochs+1):
            # Iterate over the training data.
            count = 0
            for x, y in training_data:
                # Run backpropagation to generate loss derivative matrices.
                dw1, dw2 = self.backprop(x, y)

                # Adjust weights using loss derivatives restricted by learning rate.
                self.weight_1 = self.weight_1 - lr*dw1
                self.weight_2 = self.weight_2 - lr*dw2

                # Print model progress.
                if (count == 9):
                    print("X:")
                    print(x)
                    print("Y:")
                    print(y)
                    print("Y_pred")
                    print(self.feedforward(x))
                    # print("Weight matrix 1:")
                    # print(self.weight_1)
                    # print("Weight matrix 2:")
                    # print(self.weight_2)
                count += 1
            # pdb.set_trace()


            # If there is validation data, use it to test the performance of the network.
            if validation_data:
                test_results = [(self.feedforward(x), y) for (x, y) in validation_data]
                num_correct = 0
                for y_pred, y_act in test_results:
                    # if (i == epochs):
                    #     pdb.set_trace()
                    # Round all values greater than or equal to 0.1 to 1 and the rest to 0.
                    for j in range(0, y_pred.shape[0]):
                        y_pred[j] = 1 if y_pred[j] >= 0.1 else 0

                    # Compare predicted and actual results.
                    if(torch.equal(y_pred, y_act)):
                        num_correct += 1

                print("Epoch {0}: {1} / {2}".format(i, num_correct, len(test_results)))
            
            # if (i == epochs):
            #     pdb.set_trace()


# Set our vocabulary length and context window size.
v_size = 10
cw_size = 1

# Create dataset containing 10 examples from a 10 word vocabulary. Each word is
# represented by a one-hot-encoded vector of length 10 (the size of the vocabulary).
# Within each example there are two vectors: one representing the input word, and
# another containing the words within the context. The context has been set so it
# contains the words either side of the input word in the vocabulary, wrapping
# around to the start if the input word is the first or last word in the vocabulary.
# e.g. x = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], y = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0].

# Lists to hold word vectors for input and output examples.
x_vecs = []
y_vecs = []
# Create an example using each word in the vocabulary.
for i in range(0, v_size):
    # Initialise input vector with zeros and then set ith index to 1.
    x_vec = torch.zeros(v_size)
    x_vec[i] = 1
    # Append vector to input list.
    x_vecs.append(x_vec)

    # Initialise output vector with zeros and then set the i-1th and i+1th to 1.
    y_vec = torch.zeros(v_size)
    if (i == 0):
        y_vec[v_size-1] = 1
        y_vec[i+1] = 1
    elif (i == v_size - 1):
        y_vec[i-1] = 1
        y_vec[0] = 1
    else:
        y_vec[i-1] = 1
        y_vec[i+1] = 1
    # Append vector to output list.
    y_vecs.append(y_vec)

# pdb.set_trace()
# Zip lists together into a single dataset.
data = list(zip(x_vecs, y_vecs))

# tr_data = data[0:8]
# va_data = data[8:]

# Create model instance.
sg = SkipGram(v_size, 10)

# Train model with dataset.
sg.train(data, 100, 0.01, data)
