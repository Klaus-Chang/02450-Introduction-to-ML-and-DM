import numpy as np

def adaboost_accuracy(error_rates):
    """
    Calculate the accuracy of AdaBoost given the error rates of base classifiers.
    
    :param error_rates: List of error rates for each classifier in each iteration
    :return: Accuracy of the combined classifier
    """
    T = len(error_rates)  # Number of iterations
    alpha = np.zeros(T)
    for t in range(T):
        error = error_rates[t]
        # Calculate alpha (classifier weight)
        alpha[t] = 0.5 * np.log((1 - error) / error)
    
    # Calculate the accuracy of the combined classifier
    combined_error = 0.5 * np.exp(-2 * np.sum(alpha))
    accuracy = 1 - combined_error
    
    return accuracy

# Given error rates
error_rates = [0.50, 0.75, 0.50]

# Calculate accuracy
accuracy = adaboost_accuracy(error_rates)

print(f"AdaBoost combined classifier accuracy: {accuracy:.4f}")

# Find the closest option
options = [0.25, 0.50, 0.63, 0.75]
closest_accuracy = min(options, key=lambda x: abs(x - accuracy))
print(f"Closest accuracy option: {closest_accuracy:.2f}")