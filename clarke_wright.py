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

def clarke_wright_savings(coords, demand, capacity):
    nodes = sorted(coords.keys())
    depot = nodes[0]
    customers = [n for n in nodes if n != depot]
    
    dist = {}
    for i in nodes:
        for j in nodes:
            dist[(i, j)] = calculate_distance(coords[i], coords[j])
    
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = dist[(depot, i)] + dist[(depot, j)] - dist[(i, j)]
                savings.append((-s, i, j))
    
    savings.sort()
    
    routes = {}
    for cust in customers:
        routes[cust] = [depot, cust, depot]
    
    for neg_s, i, j in savings:
        route_i = None
        route_j = None
        for cust, route in routes.items():
            if i in route:
                route_i = route
            if j in route:
                route_j = route
        
        if route_i is None or route_j is None or route_i == route_j:
            continue
        
        if (route_i[-2] == i and route_j[1] == j):
            total_demand = sum(demand[c] for c in route_i[1:-1] + route_j[1:-1])
            if total_demand <= capacity:
                new_route = route_i[:-1] + route_j[1:]
                for c in new_route[1:-1]:
                    routes[c] = new_route
    
    unique_routes = []
    seen = set()
    for route in routes.values():
        route_tuple = tuple(route)
        if route_tuple not in seen:
            seen.add(route_tuple)
            unique_routes.append(route)
    
    total_dist = 0
    for route in unique_routes:
        for i in range(len(route)-1):
            total_dist += dist[(route[i], route[i+1])]
    
    return unique_routes, total_dist

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="E-n22-k4.txt")
    args = parser.parse_args()
    
    coords, demand, capacity = read_cvrplib(args.file)
    routes, total_dist = clarke_wright_savings(coords, demand, capacity)
    
    print(f"File: {args.file}")
    print(f"Total Distance: {total_dist:.2f}")