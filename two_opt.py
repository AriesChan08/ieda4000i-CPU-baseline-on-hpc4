import math
import numpy as np
import argparse

def read_cvrplib(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    node_coord_start = lines.index("NODE_COORD_SECTION\n") + 1
    demand_start = lines.index("DEMAND_SECTION\n") + 1
    depot_start = lines.index("DEPOT_SECTION\n") + 1
    
    coords = {}
    for line in lines[node_coord_start:demand_start-1]:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0].isdigit():
            node = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            coords[node] = (x, y)
    
    demand = {}
    for line in lines[demand_start:depot_start-1]:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].isdigit():
            node = int(parts[0])
            d = int(parts[1])
            demand[node] = d
    
    capacity = 0
    for line in lines:
        if line.startswith("CAPACITY"):
            capacity = int(line.strip().split()[-1])
            break
    
    return coords, demand, capacity

def calculate_distance(coord1, coord2):
    return math.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)

def create_distance_matrix(coords):
    nodes = sorted(coords.keys())
    n = len(nodes)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = calculate_distance(coords[nodes[i]], coords[nodes[j]])
    return dist_matrix, nodes

def two_opt_swap(route, i, k):
    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
    return new_route

def two_opt(route, dist_matrix):
    improved = True
    best_distance = calculate_route_distance(route, dist_matrix)
    
    while improved:
        improved = False
        for i in range(1, len(route)-2):
            for k in range(i+1, len(route)-1):
                new_route = two_opt_swap(route, i, k)
                new_distance = calculate_route_distance(new_route, dist_matrix)
                
                if new_distance < best_distance:
                    route = new_route
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break
    
    return route, best_distance

def calculate_route_distance(route, dist_matrix):
    total = 0
    for i in range(len(route)-1):
        total += dist_matrix[route[i]][route[i+1]]
    return total

def solve_cvrp_with_two_opt(coords, demand, capacity):
    dist_matrix, nodes = create_distance_matrix(coords)
    depot = nodes.index(1)
    
    customers = [n for n in nodes if n != 1]
    routes = []
    current_route = [depot]
    current_load = 0
    
    for customer in customers:
        cust_idx = nodes.index(customer)
        cust_demand = demand[customer]
        
        if current_load + cust_demand <= capacity:
            current_route.append(cust_idx)
            current_load += cust_demand
        else:
            current_route.append(depot)
            routes.append(current_route)
            current_route = [depot, cust_idx]
            current_load = cust_demand
    
    current_route.append(depot)
    routes.append(current_route)
    
    total_distance = 0
    optimized_routes = []
    for route in routes:
        opt_route, opt_dist = two_opt(route, dist_matrix)
        optimized_routes.append(opt_route)
        total_distance += opt_dist
    
    final_routes = [[nodes[i] for i in route] for route in optimized_routes]
    
    return final_routes, total_distance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="E-n22-k4.txt")
    args = parser.parse_args()
    
    coords, demand, capacity = read_cvrplib(args.file)
    routes, total_dist = solve_cvrp_with_two_opt(coords, demand, capacity)
    
    print(f"File: {args.file}")
    print(f"Total Distance: {total_dist:.2f}")