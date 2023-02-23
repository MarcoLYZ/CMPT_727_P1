import os
import sys
import numpy as np
from collections import Counter
import random
import pdb
import math
import itertools
from itertools import product
# helpers to load data
from data_helper import load_vote_data, load_incomplete_entry, load_simulate_data, generate_q4_data

# helpers to learn and traverse the tree over attributes

# pseudocounts for uniform dirichlet prior
alpha = 0.1


# --------------------------------------------------------------------------
# Naive bayes CPT and classifier
# --------------------------------------------------------------------------


class NBCPT(object):
    '''
    NB Conditional Probability Table (CPT) for a child attribute.  Each child
    has only the class variable as a parent
    '''

    def __init__(self, A_i):
        '''
        TODO create any persistent instance variables you need that hold the
        state of the learned parameters for this CPT
            - A_i: the index of the child variable
        '''
        self.A_i = A_i
        self.count_c = [0] * 2
        self.count_A_i = np.zeros((2, 2))

    def learn(self, A, C):
        '''
        TODO
        populate any instance variables specified in __init__ to learn
        the parameters for this CPT
            - A: a 2-d numpy array where each row is a sample of assignments
            - C: a 1-d n-element numpy where the elements correspond to the
              class labels of the rows in A
        '''
        for index, row in enumerate(A):
            self.count_c[C[index]] += 1
            self.count_A_i[row[self.A_i]][C[index]] += 1

    def get_cond_prob(self, entry, c):
        ''' TODO
        return the conditional probability P(A_i|C=c) for the values
        specified in the example entry and class label c
            - entry: full assignment of variables
                e.g. entry = np.array([0,1,1]) means A_0 = 0, A_1 = 1, A_2 = 1
            - c: the class
        '''
        return (self.count_A_i[entry[self.A_i]][c] + alpha) / (self.count_c[c] + 2 * alpha)


class NBClassifier(object):
    '''
    NB classifier class specification
    '''

    def __init__(self, A_train, C_train):
        '''
        TODO create any persistent instance variables you need that hold the
        state of the trained classifier and populate them with a call to
        Suggestions for the attributes in the classifier:
            - P_c: the probabilities for the class variable C
            - cpts: a list of NBCPT objects
        '''
        self.cpts = []
        self.p_c = [0] * 2
        self._train(A_train, C_train)

    def _train(self, A_train, C_train):
        ''' TODO
        train your NB classifier with the specified data and class labels
        hint: learn the parameters for the required CPTs
            - A_train: a 2-d numpy array where each row is a sample of assignments
            - C_train: a 1-d n-element numpy where the elements correspond to
              the class labels of the rows in A
        '''
        n = len(C_train)
        classes, counts = np.unique(C_train, return_counts=True)
        for i in range(len(classes)):
            self.p_c[classes[i]] = counts[i] / n

        for i in range(A_train.shape[1]):
            cpt = NBCPT(i)
            cpt.learn(A_train, C_train)
            self.cpts.append(cpt)

    def classify(self, entry):
        ''' TODO
        return the log probabilites for class == 0 and class == 1 as a
        tuple for the given entry
        - entry: full assignment of variables
        e.g. entry = np.array([0,1,1]) means variable A_0 = 0, A_1 = 1, A_2 = 1
        NOTE this must return both the predicated label {0,1} for the class
        variable and also the log of the conditional probability of this
        assignment in a tuple, e.g. return (c_pred, logP_c_pred)
        '''
        log_probs = {}
        for c in [0, 1]:
            log_prob = np.log(self.p_c[c])
            for i in range(len(entry)):
                if entry[i] != -1:
                    log_prob += np.log(self.cpts[i].get_cond_prob(entry, c))
                else:
                    prob_now = 0
                    entry[i] = 0
                    p_i_0 = sum(self.cpts[0].count_A_i)[0]/(sum(self.cpts[0].count_A_i)[0] + sum(self.cpts[0].count_A_i)[1])
                    prob_now += (p_i_0)*self.cpts[i].get_cond_prob(entry, c)
                    entry[i] = 1
                    prob_now += (1 - p_i_0)*self.cpts[i].get_cond_prob(entry, c)
                    log_prob += np.log(prob_now)
                    entry[i] = -1
            log_probs[c] = log_prob

        pred_label = max(log_probs, key=log_probs.get)
        return (pred_label, log_probs[pred_label])

    def predict_unobserved(self, entry, index):
        ''' TODO
        Predicts P(A_index  | mid entry)
        Return a tuple of probabilities for A_index=0  and  A_index = 1
        We only use the 2nd value (P(A_index =1 |entry)) in this assignment
        '''
        probs = []
        for a_i in [0, 1]:
            entry[index] = a_i
            p_ai_given_entry = 0
            for c in [0, 1]:
                count_ai = sum(self.cpts[index].count_A_i)
                p_ai = count_ai[a_i]/sum(count_ai)
                p_c = self.p_c[c]
                log_p_c_given_ai = np.log(self.p_c[c])
                for i in range(len(entry)):
                    if entry[i] != -1:
                        log_p_c_given_ai += np.log(self.cpts[i].get_cond_prob(entry, c))
                    else:
                        prob_now = 0
                        entry[i] = 0
                        p_i_0 = sum(self.cpts[0].count_A_i)[0] / (
                                    sum(self.cpts[0].count_A_i)[0] + sum(self.cpts[0].count_A_i)[1])
                        prob_now += (p_i_0) * self.cpts[i].get_cond_prob(entry, c)
                        entry[i] = 1
                        prob_now += (1 - p_i_0) * self.cpts[i].get_cond_prob(entry, c)
                        entry[i] = -1
                        log_p_c_given_ai += np.log(prob_now)
                p_ai_given_entry += p_ai*np.exp(log_p_c_given_ai)/p_c
            entry[index] = -1
            probs.append(p_ai_given_entry)
        return tuple(probs)


# load data
'''
A_data:
n: 0
y: 1

C_data:
1: democrat
0:republican
'''
A_data, C_data = load_vote_data()


def evaluate(classifier_cls, train_subset=False, subset_size=0):
    '''
    evaluate the classifier specified by classifier_cls using 10-fold cross
    validation
    - classifier_cls: either NBClassifier or other classifiers
    - train_subset: train the classifier on a smaller subset of the training
      data
    -subset_size: the size of subset when train_subset is true
    NOTE you do *not* need to modify this function
    '''
    global A_data, C_data

    A, C = A_data, C_data

    # partition train and test set for 10 rounds
    M, N = A.shape
    tot_correct = 0
    tot_test = 0
    train_correct = 0
    train_test = 0
    step = int(M / 10 + 1)
    for holdout_round, i in enumerate(range(0, M, step)):
        # print("Holdout round: %s." % (holdout_round + 1))
        A_train = np.vstack([A[0:i, :], A[i + step:, :]])
        C_train = np.hstack([C[0:i], C[i + step:]])
        A_test = A[i: i + step, :]
        C_test = C[i: i + step]
        if train_subset:
            A_train = A_train[: subset_size, :]
            C_train = C_train[: subset_size]
        # train the classifiers
        classifier = classifier_cls(A_train, C_train)

        train_results = get_classification_results(classifier, A_train, C_train)
        # print(
        #    '  train correct {}/{}'.format(np.sum(train_results), A_train.shape[0]))
        test_results = get_classification_results(classifier, A_test, C_test)
        tot_correct += sum(test_results)
        tot_test += len(test_results)
        train_correct += sum(train_results)
        train_test += len(train_results)

    return 1. * tot_correct / tot_test, 1. * train_correct / train_test

    # score classifier on specified attributes, A, against provided labels,
    # C


def get_classification_results(classifier, A, C):
    results = []
    pp = []
    for entry, c in zip(A, C):
        c_pred, unused = classifier.classify(entry)
        results.append((c_pred == c))
        pp.append(unused)
        # print('logprobs', np.array(pp))
    return results


def evaluate_incomplete_entry(classifier_cls):
    global A_data, C_data

    # train a classifier on the full dataset
    classifier = classifier_cls(A_data, C_data)

    # load incomplete entry 1
    entry = load_incomplete_entry()

    c_pred, logP_c_pred = classifier.classify(entry)
    print("  P(C={}|A_observed) = {:2.4f}".format(c_pred, np.exp(logP_c_pred)))

    return


def predict_unobserved(classifier_cls, index=11):
    global A_data, C_data

    # train a classifier on the full dataset
    classifier = classifier_cls(A_data, C_data)
    # load incomplete entry 1
    entry = load_incomplete_entry()

    a_pred = classifier.predict_unobserved(entry, index)
    print("  P(A{}=1|A_observed) = {:2.4f}".format(index + 1, a_pred[1]))

    return


def q4_solution(classifier_cls, subset_size=400):
    global A_data, C_data
    # load the synthetic data
    if not os.path.exists('./data/q4_data'):
        generate_q4_data(4000, './data/q4_data')
    A_synthetic_data, C_synthetic_data = load_simulate_data('./data/q4_data')
    republican_count = 0
    democrat_count = 0
    republican_yes = [0] * 16
    democrat_yes = [0] * 16
    # print("Holdout round: %s." % (holdout_round + 1))
    A_train = A_synthetic_data[0:subset_size, :]
    C_train = C_synthetic_data[0:subset_size]
    # train the classifiers
    classifier = classifier_cls(A_train, C_train)

    # democrat = 1,republican = 0
    for index, entry in enumerate(A_data):
        c_pred, _ = classifier.classify(entry)
        if c_pred == 0:
            republican_count += 1
            for bill, vote in enumerate(entry):
                republican_yes[bill] += vote
        else:
            democrat_count += 1
            for bill, vote in enumerate(entry):
                democrat_yes[bill] += vote
    nonpartisan_count = 0

    for index in range(16):
        if abs(republican_yes[index]/republican_count - democrat_yes[index]/democrat_count) < 0.1:
            nonpartisan_count += 1
    return nonpartisan_count/16




def main():
    '''
    TODO modify or use the following code to evaluate your implemented
    classifiers
    Suggestions on how to use the starter code for Q2, Q3, and Q5:


    ##For Q1
    print('Naive Bayes')
    accuracy, num_examples = evaluate(NBClassifier, train_subset=False)
    print('  10-fold cross validation total test error {:2.4f} on {} '
          'examples'.format(1 - accuracy, num_examples))
    ##For Q3
    print('Naive Bayes (Small Data)')
    train_error = np.zeros(10)
    test_error = np.zeros(10)
    for x in range(10):
      accuracy, train_accuracy = evaluate(NBClassifier, train_subset=True,subset_size = (x+1)*10)
      train_error[x] = 1-train_accuracy
      test_error[x] = 1- accuracy
      print('  10-fold cross validation total test error {:2.4f} total train error {:2.4f}on {} ''examples'.format(1 - accuracy, 1- train_accuracy  ,(x+1)*10))
    print(train_error)
    print(test_error)
    ##For Q4
    print('Naive Bayes (Question 4)')
    nonpartisan_fraction = np.zeros(10)
    for x in range(10):
        nonpartisan_fraction[x] = q4_solution(NBClassifier, subset_size=(x + 1) * 400)
        print('nonpartisan fraction {:2.4f} on {} ''examples'.format(
            nonpartisan_fraction[x], (x + 1) * 400))
    print(nonpartisan_fraction)

    ##For Q5
    print('Naive Bayes Classifier on missing data')
    evaluate_incomplete_entry(NBClassifier)

    index = 11
    print('Prediting vote of A%s using NBClassifier on missing data' % (
        index + 1))
    predict_unobserved(NBClassifier, index)
   '''
    print('Naive Bayes Classifier on missing data')
    evaluate_incomplete_entry(NBClassifier)

    index = 11
    print('Prediting vote of A%s using NBClassifier on missing data' % (
            index + 1))
    predict_unobserved(NBClassifier, index)

if __name__ == '__main__':
    main()
