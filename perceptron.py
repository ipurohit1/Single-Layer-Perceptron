import random as r 
import numpy as np 
from matplotlib import pyplot as plt


def initialize_data(): 
    epochs = 5
    learning_rate = 0.1

    # each array has two features and the third element is a boolean to represent the two possible classification lables 
    # binary classification problem with True and False 
    # data_points = [ [0.1, 1.0, True], 
    #                 [0.2, 1.0, True],
    #                 [0.5, 1.5, True],
    #                 [0.8, 1.2, True],
    #                 [1.0, 2.0, True],
    #                 [3.0, 6.5, False],
    #                 [3.7, 7.0, False],
    #                 [4.4, 7.5, False],
    #                 [4.7, 8.0, False],
    #                 [5.1, 8.5, False]]

    data = [[1.0, 0.08, 0.5, 1.0], 
                     [1.0, 0.10, 1.00, 0.0],
                     [1.0, 0.26, 0.35, 1.0],
                    [1.0, 0.35, 0.95, 0.0],
                     [1.0, 0.45, 0.15, 1.0], 
                     [1.0, 0.60, 0.10, 1.0], 
                     [1.0, 0.70, 0.65, 0.0], 
                     [1.0, 0.92, 0.45, 0.0]]

    data_points = np.array([[0.08, 0.5], 
                     [0.10, 1.00], 
                     [0.26, 0.35], 
                     [0.35, 0.95], 
                     [0.45, 0.15], 
                     [0.60, 0.10], 
                     [0.70, 0.65], 
                     [0.92, 0.45]])

    # initialized to be 0 as per the perceptron training algorithm 
    weights = [0.5, -1, -1]

    return epochs, learning_rate, data, data_points, weights

def get_weighted_sum(sample, weights): 
    weighted_sum = 0

    for i in range(len(weights)): 
        weighted_sum += weights[i] * sample[i]

    return weighted_sum

def train_perceptron_weights(epochs, learning_rate, weights, data): 

    for i in range(epochs):
        print('Epoch: ' + str(i)) 
        mistakes = 0
        correct = 0

        for sample in data: 
            weighted_sum = get_weighted_sum(sample, weights)
            prediction = 1.0 if weighted_sum > 0.0 else 0.0
            print('weighted sum: ' + str(weighted_sum))
            print('prediction: ' + str(prediction))
            print

            # checking if the prediction is correct according to our training data
            if prediction == sample[-1]: 
                correct += 1
                print('Correct')

            # prediction is greater than actual classification, decrease weights
            elif prediction > sample[2]:
                mistakes += 1
                for i in range(len(weights)):
                    weights[i] = weights[i] - (learning_rate * sample[i]) 
                print('branch 2')
                # print(' '.join(map(str, weights))) 

                
            # prediction is less than actual classification, increase weights 
            else: 
                mistakes += 1
                for i in range(len(weights)): 
                    weights[i] = weights[i] + (learning_rate * sample[i]) 
                print('branch 3')
                print(' '.join(map(str, weights))) 

        if mistakes == 0: 
            print('Weights have been calculated after completing only ' + str(i + 1) +  ' epochs')
            accuracy = correct/len(data)
            return accuracy, weights
        
    print(str(epochs) + ' epochs have been completed but %s%% ' % 100 + 'accuracy has not been reached')
    accuracy = correct/len(data)
    return accuracy, weights
                
     

if __name__ == '__main__': 
    epochs, learning_rate, data, data_points, weights = initialize_data()
    model_accuracy, weights = train_perceptron_weights(epochs, learning_rate, weights, data)
    print('Model Accuracy: ' + str(model_accuracy * 100) + '\n')
    print('Weights: ')
    print(' '.join(map(str, weights))) 

    # x, y = data_points.T
    # plt.scatter(x, y)
    # plt.show()