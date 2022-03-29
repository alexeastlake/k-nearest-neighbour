import sys
import pandas as pd
import numpy as np

# Reads data files and creates training and test subsets
df = pd.read_csv(sys.argv[1], delimiter=",")
training_df = df.sample(frac = 0.8)
test_df = df.drop(training_df.index)


def main():
    test()


def test():
    correct_predictions = 0

    for test_row in test_df.itertuples():
        # Gets the predicted and actual class of the test instance
        predicted_class = knearestneighbour(test_row)
        actual_class = test_row[len(test_row) - 1]

        if predicted_class == actual_class:
            correct_predictions += 1

    # Classification accuracy
    print("Classification Accuracy: " + str((correct_predictions / len(test_df)) * 100) + "%")


def knearestneighbour(test_row):
    # k value - can be altered for differing results
    k = 3

    # Stores the ranges of the different features
    ranges = []

    # Calculates the range of the different features
    for i in range(len((training_df.columns)) - 1):
        ranges.append(training_df.iloc[:, i].max() - training_df.iloc[:, i].min())

    nearest = []
    nearest_distances = []

    for training_row in training_df.itertuples():
        # Gets the distance between the instances
        distance = calculate_distance(training_row, test_row, ranges)

        # Loops for the amount of nearest instances (k) to remember
        for i in range(k):
            if not nearest or len(nearest) < k:
                nearest.append(training_row[len(training_row) - 1])
                nearest_distances.append(distance)
                break
            elif distance < max(nearest_distances):
                nearest[nearest_distances.index(max(nearest_distances))] = training_row[len(training_row) - 1]
                nearest_distances[nearest_distances.index(max(nearest_distances))] = distance
                break

    majority = None
    majority_amount = 0

    # Gets the majority class of the nearest instances
    for i in set(nearest):
        if not majority or nearest.count(i) > majority_amount:
            majority = i
            majority_amount = nearest.count(i)

    return majority


def calculate_distance(row1, row2, ranges):
    total_before_sqrt = 0

    # Calculates total before square root
    for i in range(1, len(row1) - 1):
        value1 = row1[i]
        value2 = row2[i]

        total_before_sqrt += pow((value1 - value2), 2) / pow(ranges[i - 1], 2)

    return np.sqrt(total_before_sqrt)


if __name__ == "__main__":
    main()