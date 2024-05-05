import numpy as np

# Define input and output patterns
input_patterns = np.array([[1, -1, 1], [-1, 1, -1]])
output_patterns = np.array([[1, 1], [-1, -1]])

# Compute the weights
weights = np.dot(input_patterns.T, output_patterns)

# Recall function to retrieve output from input
def recall(input_pattern):
    return np.sign(np.dot(input_pattern, weights))

# Test recall with input patterns
test_input = np.array([1, -1, 1])
result = recall(test_input)
print("Recalled Output Pattern:", result)

# Retrieve input from the output pattern
retrieved_input = np.dot(result, weights.T)
retrieved_input = np.sign(retrieved_input)
print("Retrieved Input Pattern:", retrieved_input)
