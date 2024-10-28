import numpy as np

# Parameters
N = 40  # Total number of items
K = 3   # Total number of clusters
M1 = 5  # Number of items per cluster
M2 = 10  # Number of items per cluster
M3 = 15 # Number of items per cluster


# Generate random similarity scores
similarity_matrix = np.random.rand(N, N)
similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2  # Make it symmetric
np.fill_diagonal(similarity_matrix, 0)  # Zero diagonal

# Create input file content
file_content = f"{N} {K} {M1} {M2} {M3}\n"
for i in range(N):
    file_content += " ".join(map(str, similarity_matrix[i, i+1:])) + "\n"

# Save to file
input_file_path = "kcluster40_3_10_10_10v15.txt"
with open(input_file_path, 'w') as file:
    file.write(file_content)

input_file_path
