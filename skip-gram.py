import torch
import torch.nn.functional as fn
import pdb
import data_loader

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
    # prediction of the words within a context window of the input word. Rather than using
    # a one-hot-encoded vector representation of the input word, the function is optimised
    # to use the index of the word in the vocabulary instead.
    def feedforward(self, x_indx):
        # Multiply the input vector by the transpose of the first weight matrix to get
        # the hidden layer vector. This is equivalent to selecting the row with index equal
        # to x_indx, so we'll do this as it's far less computationally expensive.
        # h = torch.matmul(self.weight_1.transpose(0, 1), x)
        h = self.weight_1[x_indx]

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
    def backprop(self, x_indx, y_act):
        # Feed input word vector through network to generate a prediction, and store the
        # relevant intermediary vectors for use in backpropagation.
        # h = torch.matmul(self.weight_1.transpose(0, 1), x)
        h = self.weight_1[x_indx]
        u = torch.matmul(self.weight_2.transpose(0, 1), h)
        y_pred = fn.softmax(u)

        # Derivative of negative log loss (L) with respect to the second weight matrix.
        # Equates to the outer product between the hidden vector (h) and the difference
        # between the predicted and actual output vectors.
        dw2 = h.ger(y_pred - y_act)

        # Derivative of negative log loss (L) with respect to the first weight matrix.
        # Equates to the outer product between the input vector (x) and the difference
        # between the predicted and actual output vectors, multiplied by the second
        # weight matrix. The outer product with input vector will produce a matrix
        # with zeros everywhere except the row with index corresponding to the word
        # index in the vocabulary, so just return this row.
        # dw1 = x.ger(torch.matmul(self.weight_2, (y_pred - y_act)))
        dw1 = torch.matmul(self.weight_2, (y_pred - y_act))
        
        # pdb.set_trace()
        # Return the loss function derivatives with respect to the two weight matrices.
        return dw1, dw2

    # Function trains model on provided training data, consisting of pairs of input word
    # indices, and output word vectors. The output word vector should contain all the 
    # words within the context window of the input word. Training will be conducted for
    # the specified number of epochs at the specified learning rate. There is also the
    # option to provide validation data to allow training progress to be monitored.
    def train(self, training_data, epochs, lr, context=1, padding=True, validation_data=None):
        print(len(training_data))
        # Add padding equal to size of context window if required.
        if padding:
            for p in range(0, context):
                # Add padding to beginning. Padding is represented by "##" in index 1 of vocabulary.
                training_data.insert(0, 1)
                # Add padding to end.
                training_data.append(1)

                if validation_data:
                    validation_data.insert(0, 1)
                    validation_data.append(1)

        # Train for the specified number of epochs.
        for i in range(1, epochs+1):
            count = 0
            # Iterate over the training data.
            for wn in range(context, len(training_data) - context):
                x_indx = training_data[wn]

                y_indxs = []
                # Add words within the context window either side of wn.
                for cw in range(1, context + 1):
                    y_indxs.append(training_data[wn-cw])
                    y_indxs.append(training_data[wn+cw])

                y_vec = torch.zeros(self.vocab_size)
                for y_indx in y_indxs:
                    y_vec[y_indx] = 1

                # pdb.set_trace()

                # Run backpropagation to generate loss derivative matrices.
                dw1, dw2 = self.backprop(x_indx, y_vec)

                # Adjust weights using loss derivatives restricted by learning rate.
                # In W1, only the row corresponding to input word "x_indx" needs adjusting.
                self.weight_1[x_indx] = self.weight_1[x_indx] - lr*dw1
                self.weight_2 = self.weight_2 - lr*dw2

                # Print model progress.
                if (count == 9 and False):
                    print("X: {0}".format(x_indx))
                    print("Y:")
                    print(y_vec)
                    print("Y_pred")
                    print(self.feedforward(x_indx))
                    # print("Weight matrix 1:")
                    # print(self.weight_1)
                    # print("Weight matrix 2:")
                    # print(self.weight_2)
                count += 1
            # pdb.set_trace()


            # If there is validation data, use it to test the performance of the network.
            if validation_data:
                test_results = []
                for wn in range(context, len(validation_data) - context):
                    x_indx = validation_data[wn]

                    y_indxs = []
                    # Add words within the context window either side of wn.
                    for cw in range(1, context + 1):
                        y_indxs.append(validation_data[wn-cw])
                        y_indxs.append(validation_data[wn+cw])

                    y_vec = torch.zeros(self.vocab_size)
                    for y_indx in y_indxs:
                        y_vec[y_indx] = 1

                    test_results.append((self.feedforward(x_indx), y_vec))

                num_correct = 0
                for y_pred, y_act in test_results:
                    # if (i == epochs):
                    #     pdb.set_trace()
                    # Round all values greater than or equal to 0.1 to 1 and the rest to 0.
                    y_pred.gt_(0.1).type(torch.FloatTensor)
                    # for j in range(0, y_pred.shape[0]):
                    #     y_pred[j] = 1 if y_pred[j] >= 0.1 else 0

                    # Compare predicted and actual results.
                    if(torch.equal(y_pred, y_act)):
                        num_correct += 1

                print("Epoch {0}: {1} / {2}".format(i, num_correct, len(test_results)))

            # if (i == epochs):
            #     pdb.set_trace()

    # Function tests model on provided dataset and prints comparison of predicted words against
    # actual words in string format if vocabulary provided, otherwise as indices.
    def test(self, test_data, context=1, padding=True, vocab=None):
        # Add padding equal to size of context window if required.
        if padding:
            for p in range(0, context):
                # Add padding to beginning. Padding is represented by "##" in index 1 of vocabulary.
                test_data.insert(0, 1)
                # Add padding to end.
                test_data.append(1)

        test_results = []
        for wn in range(context, len(test_data) - context):
            x_indx = test_data[wn]

            y_indxs = []
            # Add words within the context window either side of wn.
            for cw in range(1, context + 1):
                y_indxs.append(test_data[wn-cw])
                y_indxs.append(test_data[wn+cw])

            y_vec = torch.zeros(self.vocab_size)
            for y_indx in y_indxs:
                y_vec[y_indx] = 1

            test_results.append((x_indx, self.feedforward(x_indx), y_vec))

        num_correct = 0
        for x_indx, y_pred, y_act in test_results:
            # Round all values greater than or equal to 0.1 to 1 and the rest to 0.
            y_pred.gt_(0.1).type(torch.FloatTensor)
            # for j in range(0, y_pred.shape[0]):
            #     y_pred[j] = 1 if y_pred[j] >= 0.1 else 0
            
            # Extract word indices from word vectors and lookup in vocabulary if available.
            # Extract indices.
            p_indxs = torch.nonzero(y_pred)
            a_indxs = torch.nonzero(y_act)

            if vocab:
                # Map to word strings.
                x_word = vocab[x_indx]
                p_words = [vocab[w] for w in p_indxs]
                a_words = [vocab[w] for w in a_indxs]

                print("x: {0}".format(x_word))
                print("y_pred:")
                print(p_words)
                print("y_act:")
                print(a_words)
            else:
                print("x: {0}".format(x_indx))
                print("y_pred:")
                print(p_indxs)
                print("y_act:")
                print(a_indxs)
                    
            # Compare predicted and actual results.
            if (torch.equal(y_pred, y_act)):
                num_correct += 1

        print("Test results: {0} / {1}".format(num_correct, len(test_results)))


# Set our vocabulary length and context window size.
vocabulary = ["my", "very", "easy", "method", "just", "speeds", "up", "naming", "planets", "."]
v_size = len(vocabulary)
cw_size = 1

# Create dataset containing 10 examples from a 10 word vocabulary. Each word is
# represented by a one-hot-encoded vector of length 10 (the size of the vocabulary).
# Within each example there is: and number representing the index of the input word in
# the vocabulary, and a one-hot-encoded vector containing the words within the context.
# The context has been set so it contains the words either side of the input word in
# the vocabulary, wrapping around to the start if the input word is the first or last
# word in the vocabulary.
# e.g. x = 2, y = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0].

# Lists to hold word vectors for input and output examples.
x_indxs = []
y_vecs = []
# Create an example using each word in the vocabulary.
for i in range(0, v_size):
    # Append index to input list.
    x_indxs.append(i)

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
data = list(zip(x_indxs, y_vecs))

# tr_data = data[0:8]
# va_data = data[8:]

dl = data_loader.IMDBDataLoader("./data/aclImdb/imdb.vocab", "./data/aclImdb/train/", "./data/aclImdb/test/", 1, 1)

tr_data = dl.ptrainex[0][2]
va_data = tr_data[:]
te_data = tr_data[:]

# Create model instance.
sg = SkipGram(len(dl.vocab), 10)

# Train model with dataset.
sg.train(training_data=tr_data, epochs=300, lr=0.005, context=2, validation_data=va_data)

# Test model.
sg.test(te_data, context=2, vocab=dl.vocab)
