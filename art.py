# 12.	Write a python program to illustrate ART neural network.

import pandas as pd
import numpy as np


def initialize_weights(input_dims, category):
    weights = np.random.uniform(size=(input_dims,))
    weights /= np.sum(weights)
    return weights


def similarity(input_pattern, weights):
    return np.minimum(input_pattern, weights).sum()


def update(imput_pattern, weights, vigilance):
    while True:
        activation = similarity(input_pattern, weights)
        if activation >= vigilance:
            return weights

        else:
            weights[np.argmax(input_pattern)] += 1
            wights /= np.sum(weights)


def art(input_patterns, vigilance):
    num_patterns, input_dims = input_patterns.shape
    categories = []

    for pattern in input_patterns:
        matched_category = None
        for category in categories:
            if similarity(pattern, category["weights"]) >= vigilance:
                matched_category = category
                break

        if matched_category == None:
            weights = initialize_weights(input_dims, len(categories))
            matched_category = {"weights": weights, "patterns": []}
            categories.append(matched_category)

        matched_category["patterns"].append(pattern)
        matched_category["weights"] = update(
            pattern, matched_category["weights"], vigilance
        )
    return categories


input_pattern = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0]])
vigilance = 0.5

categories = art(input_pattern, vigilance)

for i, category in enumerate(categories):
    print(f"categpries {i+1}:")
    print("Patterns")
    [print(pattern) for pattern in category["patterns"]]
    print("weights:")
    print(category["weights"])
    print()
