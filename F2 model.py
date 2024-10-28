import time
from mip import *

# Reading the input file
with open("kcluster40_3_10_10_10v15.txt", "r") as file:
    lines = file.readlines()

# Parsing the first line to get N, K, and Mk values
first_line = list(map(int, lines[0].strip().split()))
N = first_line[0]
K = first_line[1]
Mk = first_line[2:]

# Parsing the similarity matrix
s = np.zeros((N, N))
for i in range(1, N + 1):
    values = list(map(float, lines[i].strip().split()))
    for j, value in enumerate(values, start=i):
        s[i-1, j] = value
        s[j, i-1] = value

# Creating the model
model = Model()

# binary variable indicating whether items i and j are in the same cluster k(=1) or not (=0)
y = [[[model.add_var('y({})({})({})'.format(i, j, k), var_type=BINARY) for k in range(K)] for j in range(i + 1, N)] for i in range(N - 1)]

# binary variable indicating whether item i is in cluster k(=1) or not(=0)
x = [[model.add_var('xi({})({})'.format(i, k), var_type=BINARY) for k in range(K)] for i in range(N)]

# constraint: yijk = 1 whenever xik = 1 = xjk
for i in range(N - 1):
    for j in range(i + 1, N):
        for k in range(K):
            model += y[i][j - i - 1][k] >= x[i][k] + x[j][k] - 1

# constraint: each item i to belong to one cluster at most
for i in range(N):
    model += xsum(x[i][k] for k in range(K)) <= 1

# constraint: do not allow violation of the number of items in each cluster
for k in range(K):
    model += xsum(x[i][k] for i in range(N)) == Mk[k]
# constraint: force yijk = 1 for Mk-1 variables yijk, for fixed j and k
for j in range(N):
    for k in range(K):
        model += xsum(y[i][j - i - 1][k] for i in range(j)) + xsum(y[j][i - j - 1][k] for i in range(j + 1, N)) == (Mk[k] - 1) * x[j][k]

# objective function: maximize the sum of similarities
model.objective = maximize(
    xsum(s[i][j] * y[i][j - i - 1][k] for i in range(N - 1) for j in range(i + 1, N) for k in range(K))
)
model.write("provaF2.lp")

# Solving the model and measuring the time
start_time = time.time()
status = model.optimize(max_seconds=600)
end_time = time.time()
solution_time = end_time - start_time

# Best objective solution
best_solution = model.objective_value

# Check optimality
is_optimal = status == OptimizationStatus.OPTIMAL

# Get upper bound
upper_bound = model.objective_bound

# Print results
for i in range(N):
    for k in range(K):
        if x[i][k].x >= 0.99:
            print(f"Item {i} is in cluster {k}")

for i in range(N - 1):
    for j in range(i + 1, N):
        for k in range(K):
            if y[i][j - i - 1][k].x >= 0.99:
                print(f"Items {i} and {j} are in the same cluster {k}")

# Write to output file
output_file_path = "ResultF2.txt"
with open(output_file_path, 'a') as output_file:
    output_file.write(f"Instance file: {"kcluster40_3_10_10_10v15.txt"}  Solution: {best_solution} ({'Optimal' if is_optimal else 'Feasible'}) Solution time: {solution_time:.2f} seconds Best bound: {upper_bound}\n")