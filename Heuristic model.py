import numpy as np
import time


def read_input(file_name):
    # Reading the input file
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # Parsing the first line to get N, K, and Mk values
    first_line = list(map(int, lines[0].strip().split()))
    N = first_line[0]  # Numero di oggetti
    K = first_line[1]  # Numero di cluster
    Mk = first_line[2:]  # Dimensione di ciascun cluster

    # Parsing the similarity matrix
    s = np.zeros((N, N))
    for i in range(1, N + 1):
        values = list(map(float, lines[i].strip().split()))
        for j, value in enumerate(values, start=i):
            s[i - 1, j] = value
            s[j, i - 1] = value

    return N, K, Mk, s


def constructive_heuristic(N, K, Mk, s, time_limit=60):
    start_time = time.time()
    best_objective_value = -np.inf
    improving_swaps = 0

    # Calculate similarity sum for each object and sort in descending order
    similarity_sums = [(i, np.sum(s[i])) for i in range(N)]
    similarity_sums.sort(key=lambda x: x[1], reverse=True)

    # Initialize empty clusters
    clusters = [[] for _ in range(K)]

    # Start with the most compatible object pair (maximum sij)
    sorted_items = [i[0] for i in similarity_sums]

    # For the first cluster, select the most compatible pair
    max_similarity_pair = (-1, -1)
    max_similarity_value = -1
    for i in range(N):
        for j in range(i + 1, N):
            if s[sorted_items[i], sorted_items[j]] > max_similarity_value:
                max_similarity_value = s[sorted_items[i], sorted_items[j]]
                max_similarity_pair = (sorted_items[i], sorted_items[j])

    # Add the pair to the first cluster and remove them from the list
    clusters[0].append(max_similarity_pair[0])
    clusters[0].append(max_similarity_pair[1])
    sorted_items.remove(max_similarity_pair[0])
    sorted_items.remove(max_similarity_pair[1])

    # Add other objects that maximize the objective
    for cluster_index in range(K):
        while len(clusters[cluster_index]) < Mk[cluster_index]:
            current_time = time.time()
            if current_time - start_time >= time_limit:
                # Stop the algorithm if time has expired
                break

            best_item = None
            best_increase = -1
            for item in sorted_items:
                increase = sum(s[item][c] for c in clusters[cluster_index])
                if increase > best_increase:
                    best_increase = increase
                    best_item = item
            if best_item is not None:
                clusters[cluster_index].append(best_item)
                sorted_items.remove(best_item)

    # Improvement phase with item swapping
    for item in sorted_items:
        for cluster_index in range(K):
            for cluster_item in clusters[cluster_index]:
                current_time = time.time()
                if current_time - start_time >= time_limit:
                    # Stop the algorithm if time has expired
                    break

                current_value = sum(s[cluster_item][c] for c in clusters[cluster_index])
                new_value = sum(s[item][c] for c in clusters[cluster_index] if c != cluster_item)
                if new_value > current_value:
                    # Improving swap
                    clusters[cluster_index].remove(cluster_item)
                    clusters[cluster_index].append(item)
                    sorted_items.remove(item)
                    sorted_items.append(cluster_item)
                    improving_swaps += 1
                    break

    # Calculate the final objective function value
    objective_value = 0
    for cluster_index in range(K):
        for i in range(len(clusters[cluster_index])):
            for j in range(i + 1, len(clusters[cluster_index])):
                objective_value += s[clusters[cluster_index][i]][clusters[cluster_index][j]]

    # Check if the found solution is the best so far
    if objective_value > best_objective_value:
        best_objective_value = objective_value

    end_time = time.time()
    time_to_best = end_time - start_time

    return clusters, best_objective_value, improving_swaps, time_to_best


def write_output(file_name, instance_name, objective_value, improving_swaps, time_to_best, total_time):
    with open(file_name, 'a') as output_file:
        output_file.write(f"Instance file: {instance_name} ")
        output_file.write(f"Objective value: {objective_value} ")
        output_file.write(f"Improving swaps: {improving_swaps} ")
        output_file.write(f"Time to best: {time_to_best} seconds ")
        output_file.write(f"Total execution time: {total_time} seconds ")
        output_file.write("\n")


# Main function
def main():
    input_file_name = "kcluster40_1_20v15.txt"
    output_file_name = "Heuristicoutput.txt"
    time_limit = 600  # Time limit

    # Read data from the input file
    N, K, Mk, s = read_input(input_file_name)

    # Apply the constructive heuristic
    start_time_total = time.time()
    clusters, objective_value, improving_swaps, time_to_best = constructive_heuristic(N, K, Mk, s, time_limit)
    end_time_total = time.time()
    total_time = end_time_total - start_time_total

    # Write results to the output file
    write_output(output_file_name, input_file_name, objective_value, improving_swaps, time_to_best, total_time)


if __name__ == "__main__":
    main()
