from string import punctuation, digits
import numpy as np
import random

# Part I


def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.(+1 or -1)
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.

    -----
    def check_hinge_loss_single():
    ex_name = "Hinge loss single"

    feature_vector = np.array([1, 2])
    label, theta, theta_0 = 1, np.array([-1, 1]), -0.2
    exp_res = 1 - 0.8
    if check_real(
            ex_name, p1.hinge_loss_single,
            exp_res, feature_vector, label, theta, theta_0):
        return
    log(green("PASS"), ex_name, "")

    """
    # print(feature_vector @ theta)
    z = (feature_vector @ theta + theta_0)*label
    return 0 if z >= 1 else 1 -z

feature_vector = np.array([1, 2])
label, theta, theta_0 = 1, np.array([-1, 1]), -0.2
# print(hinge_loss_single(feature_vector, label, theta, theta_0))


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.

    def check_hinge_loss_full():
    ex_name = "Hinge loss full"

    feature_vector = np.array([[1, 2], [1, 2]])
    labels, theta, theta_0 = np.array([1, 1]), np.array([-1, 1]), -0.2
    exp_res = 1 - 0.8
    if check_real(
            ex_name, p1.hinge_loss_full,
            exp_res, feature_vector, label, theta, theta_0):
        return

    np.average(np.array( [3.40039872, 0.,         0.,         0.,         3.85457516 0.,
 0.,         3.90727499, 0.,         0.]))
    """
    z = (feature_matrix @ theta + theta_0) * labels
    # print("without norm", z)
    z[z >= 1] = np.inf
    z[z < 1] = 1 - z[z < 1]

    z[z == np.inf] = 0
    # ç("after processing", z)
    return np.average(z)

#testing - DELETE
# feature_matrix = np.array([[1, 2], [1, 2]])
# labels, theta, theta_0 = np.array([1, 1]), np.array([-1, 1]), -0.2
# hinge_loss_full(feature_matrix, labels, theta, theta_0)

def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.

    if (feature_vector @ current_theta + current_theta_0) * label <= 0:
        current_theta += (feature_vector * label)
        current_theta_0 += label

    return current_theta, current_theta_0

    """
    if (feature_vector @ current_theta + current_theta_0 ) * label <= 0:
        current_theta += (feature_vector * label)
        current_theta_0 += label
    return current_theta, current_theta_0

# def perceptron_check():
#     feature_vector = np.array([1, 2])
#     label, theta, theta_0 = 1, np.array([-1, 1]), -1.5
#     exp_res = (np.array([0, 3]), -0.5)
#
#     res = perceptron_single_step_update(feature_vector, label, theta, theta_0)
#     print (res, exp_res)
#
# perceptron_check()

def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    # set initial values
    theta, theta_0 = np.zeros(feature_matrix.shape[1]), 0
    # outer loop
    for t in range(T):
        # loop over each row
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i,:], labels[i], theta, theta_0)
            # print(theta,theta_0)
    return theta, theta_0
#
# def check_final():
#     feature_matrix = np.array([[1, 2]])
#     labels = np.array([1])
#     T = 2
#     exp_res = (np.array([1, 2]), 1)
#     res = perceptron(feature_matrix, labels, T)
#     return res, exp_res
# check_final()

def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    # Tip:tracking a sum through loops is simple.
    # set initial values
    theta, theta_0 = np.zeros(feature_matrix.shape[1]), 0
    theta_sum = np.zeros(feature_matrix.shape[1])
    theta_0_sum = 0
    # outer loop
    for t in range(T):
        # loop over each row
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i, :], labels[i], theta, theta_0)
            theta_sum += theta
            theta_0_sum += theta_0
    # n is the number of feature vectors
    return theta_sum / (feature_matrix.shape[0] * T), theta_0_sum /(feature_matrix.shape[0] * T)

### PEGASOS
def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.

            decision = y[it] * weights @ x[it].T
        if decision < 1:
            weights = (1 - step*lam) * weights + step*y[it]*x[it]
        else:
            weights = (1 - step*lam) * weights
    """
    #  take care not to penalize the magnitude of bias (beta_0)

    # condition:  label * prediction smaller than or equal to 1
    # "you will need to adapt this update rule to add a bias term)
    if label * (current_theta @ feature_vector + current_theta_0) <= 1:
        current_theta = ( 1 - eta*L)*current_theta + eta*label*feature_vector
        current_theta_0 += eta * label
    else: # here is the mistake, in the current theta
        current_theta = (1 - eta * L) * current_theta
        # update theta_0 ? no

    return current_theta, current_theta_0


def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lambda value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    # --------
    test
    feature_matrix = np.array([[1, 1], [1, 1]])
    labels = np.array([1, 1])
    T = 1
    L = 1
    exp_res = (np.array([1-1/np.sqrt(2), 1-1/np.sqrt(2)]), 1)
    """

    # initialize parameters to 0
    theta, theta_0 = np.zeros(feature_matrix.shape[1]), 0

    loop_counter = 1 # counter number of updates we have done so far. (1,nT) inclusive

    # outer loop
    for t in range(T):
        # loop over each row
        for i in get_order(feature_matrix.shape[0]): # loop over row, corresponding to feature vector
            current_feature_vector = feature_matrix[i,:]
            current_label = labels[i]
            eta = 1/np.sqrt(loop_counter) # learning rate

            # run single_step function at each iteration; update parameters
            theta, theta_0 = pegasos_single_step_update(current_feature_vector, current_label, L, eta, theta, theta_0)

            # print(loop_counter)
            loop_counter += 1 # update after each of the n*T iterations.

    return theta, theta_0

# def check_pegasos():
#     print ("checking pegasos...")
#     feature_matrix = np.array([[1, 1], [1, 1]])
#     labels = np.array([1, 1])
#     T = 1
#     L = 1
#     res = pegasos(feature_matrix, labels, T, L)
#     exp_res = (np.array([1 - 1 / np.sqrt(2), 1 - 1 / np.sqrt(2)]), 1)
#     print (res, exp_res)
#
# check_pegasos()
# # ----------------------
# Part II


def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    #Tip:: As in previous exercises, when x is a float, “x = 0" should be checked with abs(x) < e (small number)
    epsilon = 0.00001
    out = feature_matrix @ theta + theta_0
    out[out > epsilon] = 1
    out[out <= epsilon] = -1
    return out


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args (6):
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the validation
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """

    # training model based on classifier (e.g. Pegasos, Perceptron)
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    train_pred_labels = classify(train_feature_matrix, theta, theta_0)
    train_accuracy = accuracy(train_pred_labels, train_labels)

    # calculating accuracy on new data
    val_pred_labels = classify(val_feature_matrix, theta, theta_0)
    val_accuracy = accuracy(val_pred_labels, val_labels)

    return train_accuracy, val_accuracy

def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()

def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()

######## 9. Feature engineering. I.e. what to look at? (and what not to look at)
def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9

    original code:
        dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary
    """
    # Your code here


    # loading stopwords: auxiliary function
    stopwords = load_txt("stopwords.txt")

    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary and word not in stopwords: # additional constraint to avoid those words in dict
                dictionary[word] = len(dictionary)
    # print (dictionary)
    return dictionary


def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9:
      -use the same learning algorithm and the same feature as the last problem.
      - However, when you compute the feature vector of a word,
      use its count in each document rather than a binary indicator.

    CONCLUSION: if we consider the count, the algorithm is actually WORST.


    original code:
     num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1
    return feature_matrix

    """

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        if i ==1:
            print(word_list)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] += 1 # if we consider the count, the algorithm is actually WORST
    print(f"total words {np.sum(feature_matrix)}")
    return feature_matrix


def load_txt(filename):
    # opening the file in read mode
    my_file = open(filename, "r")

    # reading the file
    data = my_file.read()

    # replacing end of line('/n') with ' ' and
    # splitting the text it further when '.' is seen.
    data_into_list = data.replace('\n', ' ').split()

    # make sure to close file
    my_file.close()

    return data_into_list




