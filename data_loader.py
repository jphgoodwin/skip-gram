import pdb
import os
import re

class IMDBDataLoader():
    def __init__(self, vocab_path, train_path, test_path, ntrain, ntest):
        # Declare instance variables.
        self.ntrain = None         # Number of training examples to load.
        self.ntest = None          # Number of test examples to load.
        self.vocabPath = None      # Path to vocab file.
        self.vocab = None          # Loaded vocabulary.
        self.trainPath = None      # Path to train/ directory.
        self.ptrainex = None       # Loaded positive training examples.
        self.ntrainex = None       # Loaded negative training examples.
        self.utrainex = None       # Loaded unsupervised training examples.
        self.testPath = None       # Path to test/ directory.
        self.ptestex = None        # Loaded positive test examples.
        self.ntestex = None        # Loaded negative test examples.

        # Set number of training and test examples to load.
        self.ntrain = ntrain
        self.ntest = ntest

        # Load vocab.
        self.loadVocab(vocab_path)

        # Load training data.
        self.loadTrainingData(train_path)

        # Load test data.
        self.loadTestData(test_path)
    
    def loadVocab(self, path):
        # Store path to vocab file.
        self.vocabPath = path

        # Open file for reading.
        f = open(path, "r")

        # Initialise vocabulary with unknown and padding symbols.
        self.vocab = ["unk", "##"]

        # Add numbers to vocabulary.
        for n in range(0, 10):
            self.vocab.append(n)

        # Add punctuation to vocabulary.
        self.vocab.extend([",", ".", ":", ";"])

        # Extract vocabulary from file.
        for line in f:
            self.vocab.append(line[:-1])

        # Close file.
        f.close()

    def loadTrainingData(self, path):
        print("Loading training data.")

        # Store path to training data directory.
        self.trainPath = path

        # Get file lists for positive, negative and unsupervised directories.
        flp = os.listdir(path + "pos")
        fln = os.listdir(path + "neg")
        flu = os.listdir(path + "unsup")

        # Positive, negative and unsupervised training example lists.
        ptel = []
        ntel = []
        utel = []

        # Load data from files in positive directory.
        for i in range(0, min(self.ntrain, len(flp))):
            # Get filename from list.
            f = flp[i]

            # Extract example number and rating from filename.
            num = f.split("_")[0]
            rating = f.split("_")[1].split(".")[0]

            # Open file and extract contents.
            ef = open(path + "pos/" + f, "r")            
            # Word index list.
            wilist = []
            # Split file contents into words and iterate through them.
            for w in ef.read().split():
                # Split the word on any punctuation (eg ",").
                swl = self.splitPunctuation(w)
                # Iterate over subword list.
                for sw in swl:
                    try:
                        # Try and get index for word in vocabulary.
                        indx = self.vocab.index(sw.lower())
                        # Apped to word index list.
                        wilist.append(indx)
                    except:
                        # Word isn't in vocabulary so append unk index (0).
                        # print('{0} not in vocabulary.'.format(sw.lower()))
                        wilist.append(0)

            # Append tuple of example number, rating, and word list to example list.
            ptel.append((num, rating, wilist))

        # Load data from files in negative directory.
        for i in range(0, min(self.ntrain, len(fln))):
            # Get filename from list.
            f = fln[i]

            # Extract example number and rating from filename.
            num = f.split("_")[0]
            rating = f.split("_")[1].split(".")[0]

            # Open file and extract contents.
            ef = open(path + "neg/" + f, "r")            
            # Word index list.
            wilist = []
            # Split file contents into words and iterate through them.
            for w in ef.read().split():
                # Split the word on any punctuation (eg ",").
                swl = self.splitPunctuation(w)
                # Iterate over subword list.
                for sw in swl:
                    try:
                        # Try and get index for word in vocabulary.
                        indx = self.vocab.index(sw.lower())
                        # Apped to word index list.
                        wilist.append(indx)
                    except:
                        # Word isn't in vocabulary so append unk index (0).
                        # print('{0} not in vocabulary.'.format(sw.lower()))
                        wilist.append(0)

            # Append tuple of example number, rating, and word list to example list.
            ntel.append((num, rating, wilist))
        
        # Load data from files in unsupervised directory.
        for i in range(0, min(self.ntrain, len(flu))):
            # Get filename from list.
            f = flu[i]

            # Extract example number and rating from filename.
            num = f.split("_")[0]
            rating = f.split("_")[1].split(".")[0]

            # Open file and extract contents.
            ef = open(path + "unsup/" + f, "r")            
            # Word index list.
            wilist = []
            # Split file contents into words and iterate through them.
            for w in ef.read().split():
                # Split the word on any punctuation (eg ",").
                swl = self.splitPunctuation(w)
                # Iterate over subword list.
                for sw in swl:
                    try:
                        # Try and get index for word in vocabulary.
                        indx = self.vocab.index(sw.lower())
                        # Apped to word index list.
                        wilist.append(indx)
                    except:
                        # Word isn't in vocabulary so append unk index (0).
                        # print('{0} not in vocabulary.'.format(sw.lower()))
                        wilist.append(0)
           
            # Append tuple of example number, rating, and word list to example list.
            utel.append((num, rating, wilist))

        # Store loaded examples.
        self.ptrainex = ptel
        self.ntrainex = ntel
        self.utrainex = utel

        # Print index and string values for positive example list.
        # print("Positive example list:")
        # for pte in ptel:
        #     wlist = [(i, self.vocab[i]) for i in pte[2]]
        #     print(wlist)

        # Print index and string values for negative example list.
        # print("Negative example list:")
        # for nte in ntel:
        #     wlist = [(i, self.vocab[i]) for i in nte[2]]
        #     print(wlist)

        # Print index and string values for unsupervised example list.
        # print("Unsupervised example list:")
        # for ute in utel:
        #     wlist = [(i, self.vocab[i]) for i in ute[2]]
        #     print(wlist)


    def loadTestData(self, path):
        print("Loading test data.")

        # Store path to test data directory.
        self.testPath = path

        # Get file lists for positive and negative directories.
        flp = os.listdir(path + "pos")
        fln = os.listdir(path + "neg")

        # Positive and negative test example lists.
        ptel = []
        ntel = []

        # Load data from files in positive directory.
        for i in range(0, min(self.ntest, len(flp))):
            # Get filename from list.
            f = flp[i]

            # Extract example number and rating from filename.
            num = f.split("_")[0]
            rating = f.split("_")[1].split(".")[0]

            # Open file and extract contents.
            ef = open(path + "pos/" + f, "r")
            # Word index list.
            wilist = []
            # Split file contents into words and iterate through them.
            for w in ef.read().split():
                # Split the word on any punctuation (eg ",").
                swl = self.splitPunctuation(w)
                # Iterate over subword list.
                for sw in swl:
                    try:
                        # Try and get index for word in vocabulary.
                        indx = self.vocab.index(sw.lower())
                        # Apped to word index list.
                        wilist.append(indx)
                    except:
                        # Word isn't in vocabulary so append unk index (0).
                        # print('{0} not in vocabulary.'.format(sw.lower()))
                        wilist.append(0)

            
            # Append tuple of example number, rating, and word list to example list.
            ptel.append((num, rating, wilist))

        # Load data from files in negative directory.
        for i in range(0, min(self.ntest, len(fln))):
            # Get filename from list.
            f = fln[i]

            # Extract example number and rating from filename.
            num = f.split("_")[0]
            rating = f.split("_")[1].split(".")[0]

            # Open file and extract contents.
            ef = open(path + "neg/" + f, "r")
            # Word index list.
            wilist = []
            # Split file contents into words and iterate through them.
            for w in ef.read().split():
                # Split the word on any punctuation (eg ",").
                swl = self.splitPunctuation(w)
                # Iterate over subword list.
                for sw in swl:
                    try:
                        # Try and get index for word in vocabulary.
                        indx = self.vocab.index(sw.lower())
                        # Apped to word index list.
                        wilist.append(indx)
                    except:
                        # Word isn't in vocabulary so append unk index (0).
                        # print('{0} not in vocabulary.'.format(sw.lower()))
                        wilist.append(0)
            
            # Append tuple of example number, rating, and word list to example list.
            ntel.append((num, rating, wilist))

        # Store loaded examples.
        self.ptestex = ptel
        self.ntestex = ntel

        # Print index and string values for positive example list.
        # print("Positive example list:")
        # for pte in ptel:
        #     wlist = [(i, self.vocab[i]) for i in pte[2]]
        #     print(wlist)

        # Print index and string values for negative example list.
        # print("Negative example list:")
        # for nte in ntel:
        #     wlist = [(i, self.vocab[i]) for i in nte[2]]
        #     print(wlist)


    # Function splits the input word on any punctuation and returns a subword list.
    # This list may contain only one word if no punctuation was found.
    def splitPunctuation(self, word):
        # If word is a single character then return this in a list.
        if (len(word) == 1): 
            return [word]

        # Search for punctuation characters in word.
        x = re.search("[,.!?]", word)

        # If found, split word on punctuation.
        if (x):
            # Get index of punctuation character in word.
            i = x.span()[0]

            # If it is the first character then split into two parts and recursively call to right.
            if (i == 0):
                w1 = [word[i]]
                w2 = self.splitPunctuation(word[i+1:])
                return w1 + w2

            # If it is the last character then split word into two parts.
            if (i == len(word)-1):
                return [word[0:i], word[i]]

            # Otherwise, split word into three parts and recusively call word to right.
            w1 = [word[0:i]]
            w2 = [word[i]]
            w3 = self.splitPunctuation(word[i+1:])
            return w1 + w2 + w3

        # Otherwise return word in a list.
        return [word]

# dl = IMDBDataLoader("./data/aclImdb/imdb.vocab", "./data/aclImdb/train/", "./data/aclImdb/test/", 1, 1)
