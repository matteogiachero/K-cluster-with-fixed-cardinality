import numpy as np
import time
import random


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


def constructive_heuristic(N, K, Mk, s, time_limit=600):
    start_time = time.time()

    # Calculate similarity sum for each object and sort in descending order
    similarity_sums = [(i, np.sum(s[i])) for i in range(N)]
    similarity_sums.sort(key=lambda x: x[1], reverse=True)

    # Initialize empty clusters
    clusters = [[] for _ in range(K)]

    # Sort objects by similarity sum
    sorted_items = [i[0] for i in similarity_sums]

    # Select the most compatible object pairs to start clusters
    pairs_used = set()
    for cluster_index in range(K):
        max_similarity_pair = (-1, -1)
        max_similarity_value = -1
        for i in range(len(sorted_items)):
            for j in range(i + 1, len(sorted_items)):
                if sorted_items[i] not in pairs_used and sorted_items[j] not in pairs_used:
                    if s[sorted_items[i], sorted_items[j]] > max_similarity_value:
                        max_similarity_value = s[sorted_items[i], sorted_items[j]]
                        max_similarity_pair = (sorted_items[i], sorted_items[j])

        clusters[cluster_index].append(max_similarity_pair[0])
        clusters[cluster_index].append(max_similarity_pair[1])
        pairs_used.add(max_similarity_pair[0])
        pairs_used.add(max_similarity_pair[1])

    # Remove used pairs from the list of objects
    sorted_items = [item for item in sorted_items if item not in pairs_used]

    # Add other objects that maximize the objective
    for cluster_index in range(K):
        while len(clusters[cluster_index]) < Mk[cluster_index]:
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

    # Swaps within clusters
    cluster_swaps = 0
    outside_swaps = 0
    total_swaps = 0
    best_objective_value = compute_objective(clusters, s)
    time_to_best = 0

    while time.time() - start_time < time_limit:
        improvement_found = False
        for _ in range(100):  # Perform 100 random swap attempts between clusters
            cluster_a, cluster_b = random.sample(range(K), 2)
            if clusters[cluster_a] and clusters[cluster_b]:
                item_a = random.choice(clusters[cluster_a])
                item_b = random.choice(clusters[cluster_b])

                # Evaluate if the swap improves the objective function
                current_value_a = sum(s[item_a][c] for c in clusters[cluster_a] if c != item_a)
                current_value_b = sum(s[item_b][c] for c in clusters[cluster_b] if c != item_b)

                new_value_a = sum(s[item_b][c] for c in clusters[cluster_a])
                new_value_b = sum(s[item_a][c] for c in clusters[cluster_b])

                if new_value_a + new_value_b > current_value_a + current_value_b:
                    # Improving swap between clusters
                    clusters[cluster_a].remove(item_a)
                    clusters[cluster_a].append(item_b)
                    clusters[cluster_b].remove(item_b)
                    clusters[cluster_b].append(item_a)
                    cluster_swaps += 1
                    total_swaps += 1
                    improvement_found = True

        # Update the best objective function value
        current_objective_value = compute_objective(clusters, s)
        if current_objective_value > best_objective_value:
            best_objective_value = current_objective_value
            time_to_best = time.time() - start_time

        # Stop if no improvements found after many attempts
        if not improvement_found:
            break

    # Swaps with objects outside clusters
    while time.time() - start_time < time_limit:
        improvement_found = False
        for item in sorted_items:
            for cluster_index in range(K):
                for cluster_item in clusters[cluster_index]:
                    current_value = sum(s[cluster_item][c] for c in clusters[cluster_index])
                    new_value = sum(s[item][c] for c in clusters[cluster_index] if c != cluster_item)
                    if new_value > current_value:
                        # Improving swap with objects outside clusters
                        clusters[cluster_index].remove(cluster_item)
                        clusters[cluster_index].append(item)
                        sorted_items.remove(item)
                        sorted_items.append(cluster_item)
                        outside_swaps += 1
                        total_swaps += 1
                        improvement_found = True
                        break
                if improvement_found:
                    break
            if improvement_found:
                break

        # Update the best objective function value
        current_objective_value = compute_objective(clusters, s)
        if current_objective_value > best_objective_value:
            best_objective_value = current_objective_value
            time_to_best = time.time() - start_time

        # Exit if no improvements are found
        if not improvement_found:
            break

    # Total execution time
    total_time = time.time() - start_time

    return clusters, best_objective_value, cluster_swaps, outside_swaps, total_swaps, time_to_best, total_time


def compute_objective(clusters, s):
    objective_value = 0
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                objective_value += s[cluster[i]][cluster[j]]
    return objective_value


def write_output(file_name, instance_name, objective_value, cluster_swaps, outside_swaps, total_swaps, time_to_best, total_time):
    with open(file_name, 'a') as output_file:
        output_file.write(f"Instance file: {instance_name} ")
        output_file.write(f"Objective value: {objective_value:} ")
        output_file.write(f"Improving swaps (within clusters): {cluster_swaps} ")
        output_file.write(f"Improving swaps (outside clusters): {outside_swaps} ")
        output_file.write(f"Total swaps (including between clusters): {total_swaps} ")
        output_file.write(f"Time to best: {time_to_best:} seconds ")
        output_file.write(f"Total execution time: {total_time:} seconds ")
        output_file.write("\n")


# Main function
def main():
    input_file_name = "kcluster40_3_10_10_10v15.txt"  # Cambia percorso se necessario
    output_file_name = "Heuristicoutput.txt"

    # Read data from the input file
    N, K, Mk, s = read_input(input_file_name)

    # Apply the constructive heuristic with a maximum time of 600 seconds
    clusters, objective_value, cluster_swaps, outside_swaps, total_swaps, time_to_best, total_time = constructive_heuristic(N, K, Mk,
                                                                                                               s,
                                                                                                               time_limit=600)

    # Write results to the output file
    write_output(output_file_name, input_file_name, objective_value, cluster_swaps, outside_swaps, total_swaps, time_to_best,
                 total_time)


if __name__ == "__main__":
    main()
