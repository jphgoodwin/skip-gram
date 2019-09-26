import torch
import torch.nn.functional as fn
import pdb
import data_loader
import os

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
        y = fn.softmax(u, dim=0)

        # Return the resultant vector.
        return y

    # Function performs backpropagation of negative log loss between expected and actual
    # values through the weights of the network. Returns the derivative of the loss
    # function with respect the two weight matrices.
    def backprop(self, x_indxs, y_act):
        # Feed input word vector through network to generate a prediction, and store the
        # relevant intermediary vectors for use in backpropagation.
        # h = torch.matmul(self.weight_1.transpose(0, 1), x)
        h = torch.index_select(self.weight_1, 0, x_indxs)
        u = torch.matmul(h, self.weight_2)
        y_pred = fn.softmax(u, dim=1)

        # Derivative of negative log loss (L) with respect to the second weight matrix.
        # Equates to the outer product between the hidden vector (h) and the difference
        # between the predicted and actual output vectors.
        dw2 = torch.matmul(h.transpose(0, 1), (y_pred - y_act))

        # Derivative of negative log loss (L) with respect to the first weight matrix.
        # Equates to the outer product between the input vector (x) and the difference
        # between the predicted and actual output vectors, multiplied by the second
        # weight matrix. The outer product with input vector will produce a matrix
        # with zeros everywhere except the row with index corresponding to the word
        # index in the vocabulary, so just return this row.
        # dw1 = x.ger(torch.matmul(self.weight_2, (y_pred - y_act)))
        dw1 = torch.matmul((y_pred - y_act), self.weight_2.transpose(0, 1))
        
        # Return the loss function derivatives with respect to the two weight matrices.
        return dw1, dw2

    # Function trains model on provided training data, consisting of a list of word indexes
    # into a vocabulary, over which it passes a context window to generate individual examples.
    # Within each example there is a central word which serves as input, and the words either
    # side within the context window are put together into a single one-hot-encoded vector
    # representing the target output. Training will be conducted for the given number of
    # epochs, at the specified learning rate. The batch size determines how many examples will
    # be extracted and run in parallel. The caller may specify a context window size (default 1)
    # and whether to pad the input list (default True). There is also the option to provide
    # validation data to allow training progress to be monitored.
    def train(self, training_data, epochs, bs, lr, context=1, padding=True, validation_data=None):
        # Add padding equal to size of context window if required.
        if padding:
            for p in range(0, context):
                # Add padding to beginning. Padding is represented by "##" in index 1 of vocabulary.
                training_data.insert(0, 1)
                # Add padding to end.
                training_data.append(1)
                
                # Add padding to validation data if present.
                if validation_data:
                    validation_data.insert(0, 1)
                    validation_data.append(1)

        # Train for the specified number of epochs.
        for i in range(1, epochs+1):
            # Break example range down into batches, accounting for the required context window
            # before the first and after the last examples. If no padding has been added, this will
            # mean a context window worth of examples at each end will be excluded.
            batches = [range(wn, min(wn+bs, len(training_data) - context))
                    for wn in range(context, len(training_data) - context, bs)]

            # Iterate over the batch ranges.
            for bn in batches:
                # Extract training examples from within batch range to use as input, and store in a vector.
                x_indxs = torch.tensor([training_data[wn] for wn in bn])

                # Create an array of one-hot-encoded vectors representing the context window for each
                # example in the batch.
                y_vecs = []
                for wn in bn:
                    # Extract the context word indices either side of the current word.
                    y_indxs = []
                    for cw in range(1, context+1):
                        y_indxs.append(training_data[wn-cw])
                        y_indxs.append(training_data[wn+cw])

                    # Convert these word indices into a one-hot-encoded vector.
                    y_vec = torch.zeros(self.vocab_size)
                    for y_indx in y_indxs:
                        y_vec[y_indx] = 1

                    # Append vector to list.
                    y_vecs.append(y_vec)

                # Stack context vectors in a 2D tensor.
                y_vecs = torch.stack(y_vecs)

                # Run backpropagation to generate loss derivative matrices.
                dw1, dw2 = self.backprop(x_indxs, y_vecs)

                # Adjust weights using loss derivatives restricted by learning rate.
                # In W1, only the rows corresponding to input words in x_indxs need adjusting.
                self.weight_1.index_add_(0, x_indxs, (-lr*dw1))
                self.weight_2 = self.weight_2 - lr*dw2

            # If there is validation data, use it to test the performance of the network.
            if validation_data:
                test_results = []
                # Iterate over validation data.
                for wn in range(context, len(validation_data) - context):
                    # Extract the input word from current position wn.
                    x_indx = validation_data[wn]

                    # Form output word list from words either side of current position wn.
                    y_indxs = []
                    for cw in range(1, context + 1):
                        y_indxs.append(validation_data[wn-cw])
                        y_indxs.append(validation_data[wn+cw])

                    # Covert the output word list into a one-hot-encoded vector.
                    y_vec = torch.zeros(self.vocab_size)
                    for y_indx in y_indxs:
                        y_vec[y_indx] = 1

                    # Add tuple of predicted and actual output vectors to test_results.
                    test_results.append((self.feedforward(x_indx), y_vec))

                num_correct = 0
                # Iterate over test_results and compare predicted to actual output, counting the
                # number that match.
                for y_pred, y_act in test_results:
                    # Round all values greater than or equal to 0.1 to 1 and the rest to 0.
                    y_pred.gt_(0.1).type(torch.FloatTensor)

                    # Compare predicted and actual results.
                    if(torch.equal(y_pred, y_act)):
                        num_correct += 1

                # Print results.
                print("Epoch {0}: {1} / {2}".format(i, num_correct, len(test_results)))


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
        # Iterate over test data.
        for wn in range(context, len(test_data) - context):
            # Extract the input word from current position wn.
            x_indx = test_data[wn]

            # Form output word list from words either side of current position wn.
            y_indxs = []
            for cw in range(1, context + 1):
                y_indxs.append(test_data[wn-cw])
                y_indxs.append(test_data[wn+cw])

            # Covert the output word list into a one-hot-encoded vector.
            y_vec = torch.zeros(self.vocab_size)
            for y_indx in y_indxs:
                y_vec[y_indx] = 1

            # Add tuple of predicted and actual output vectors to test_results.
            test_results.append((x_indx, self.feedforward(x_indx), y_vec))

        num_correct = 0
        # Iterate over test_results and compare predicted to actual output, counting the
        # number that match.
        for x_indx, y_pred, y_act in test_results:
            # Round all values greater than or equal to 0.1 to 1 and the rest to 0.
            y_pred.gt_(0.1).type(torch.FloatTensor)
            
            # Extract word indices from word vectors and lookup in vocabulary if available.
            # Extract indices.
            p_indxs = torch.nonzero(y_pred)
            a_indxs = torch.nonzero(y_act)

            if vocab:
                # Map to word strings.
                x_word = vocab[x_indx]
                p_words = [vocab[w] for w in p_indxs]
                a_words = [vocab[w] for w in a_indxs]

                # Print words.
                print("x: {0}, y_pred: {1}, y_act: {2}".format(x_word, p_words, a_words))
                # print("y_pred:")
                # print(p_words)
                # print("y_act:")
                # print(a_words)
            else:
                # Print indices.
                print("x: {0}".format(x_indx))
                print("y_pred:")
                print(p_indxs)
                print("y_act:")
                print(a_indxs)
                    
            # Compare predicted and actual results.
            if (torch.equal(y_pred, y_act)):
                num_correct += 1

        # Print results.
        print("Test results: {0} / {1}".format(num_correct, len(test_results)))

    # Saves model parameters (weight matrices) to directory with name model_name.
    def saveModel(self, model_name):
        try:
            # See if model_name directory already exists.
            os.listdir("./models/" + model_name)
        except FileNotFoundError:
            # If not, make it.
            os.makedirs("./models/" + model_name)
        
        # Save the weight matrices to files within the model_name directory.
        torch.save(self.weight_1, "./models/" + model_name + "/w1.pt")
        torch.save(self.weight_2, "./models/" + model_name + "/w2.pt")

    # Loads model parameters (weight matrices) from directory with model_name.
    def loadModel(self, model_name):
        try:
            # Load the saved weight matrices into our model instance.
            self.weight_1 = torch.load("./models/" + model_name + "/w1.pt")
            self.weight_2 = torch.load("./models/" + model_name + "/w2.pt")
        except FileNotFoundError:
            # Alert user if files don't exist.
            print("Saved file not found.")


# Load data from IMDB dataset.
dl = data_loader.IMDBDataLoader("./data/aclImdb/imdb.vocab", "./data/aclImdb/train/", "./data/aclImdb/test/", 1, 1)

# Extract training data as list of words, concatenating examples together.
tr_data = []
for ex in dl.ptrainex:
    tr_data.extend(ex[2])
va_data = tr_data[:]
te_data = tr_data[:]

# Create model instance.
sg = SkipGram(len(dl.vocab), 10)

sg.loadModel("model_2")

# Train model with dataset.
# sg.train(training_data=tr_data, epochs=100, bs=16, lr=0.005, context=1, validation_data=va_data)

# Save model.
# sg.saveModel("model_2")

# Test model.
sg.test(te_data, context=1, vocab=dl.vocab)
