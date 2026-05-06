import math
import random
import numpy as np
from functools import reduce
import sys
import getopt

alfa = 2
beta = 5
sigm = 3
ro = 0.8
th = 80
fileName = "E-n22-k4.txt"
iterations = 1000
ants = 22

def read_cvrplib(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    node_coord_start = lines.index("NODE_COORD_SECTION\n") + 1
    demand_start = lines.index("DEMAND_SECTION\n") + 1
    depot_start = lines.index("DEPOT_SECTION\n") + 1
    eof = lines.index("EOF\n") if "EOF\n" in lines else len(lines)
    
    graph = {}
    for line in lines[node_coord_start:demand_start-1]:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0].isdigit():
            node = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            graph[node] = (x, y)
    
    demand = {}
    for line in lines[demand_start:depot_start-1]:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].isdigit():
            node = int(parts[0])
            d = int(parts[1])
            demand[node] = d
    
    capacityLimit = 0
    for line in lines:
        if line.startswith("CAPACITY"):
            capacityLimit = int(line.strip().split()[-1])
            break
    
    optimalValue = None
    for line in lines:
        if "Optimal value" in line or "Best value" in line:
            try:
                optimalValue = int(''.join(filter(str.isdigit, line)))
            except:
                pass
    
    if "E-n22-k4" in file_path and optimalValue is None:
        optimalValue = 375
        
    return capacityLimit, graph, demand, optimalValue

def generateGraph():
    capacityLimit, graph, demand, optimalValue = read_cvrplib(fileName)
    vertices = list(graph.keys())
    vertices.remove(1)
    
    edges = { (min(a,b),max(a,b)) : np.sqrt((graph[a][0]-graph[b][0])**2 + (graph[a][1]-graph[b][1])**2) 
              for a in graph.keys() for b in graph.keys() if a != b }
    feromones = { (min(a,b),max(a,b)) : 1 for a in graph.keys() for b in graph.keys() if a != b }
    
    return vertices, edges, capacityLimit, demand, feromones, optimalValue

def solutionOfOneAnt(vertices, edges, capacityLimit, demand, feromones):
    solution = list()
    vertices = vertices.copy()
    
    while(len(vertices)!=0):
        path = list()
        city = np.random.choice(vertices)
        capacity = capacityLimit - demand[city]
        path.append(city)
        vertices.remove(city)
        
        while(len(vertices)!=0):
            probabilities = list(map(lambda x: ((feromones[(min(x,city), max(x,city))])**alfa)*((1/edges[(min(x,city), max(x,city))])**beta), vertices))
            probabilities = probabilities/np.sum(probabilities)
            city = np.random.choice(vertices, p=probabilities)
            capacity = capacity - demand[city]
            
            if(capacity>0):
                path.append(city)
                vertices.remove(city)
            else:
                break
                
        solution.append(path)
    return solution

def rateSolution(solution, edges):
    s = 0
    for i in solution:
        a = 1
        for j in i:
            b = j
            s = s + edges[(min(a,b), max(a,b))]
            a = b
        b = 1
        s = s + edges[(min(a,b), max(a,b))]
    return s

def updateFeromone(feromones, solutions, bestSolution):
    Lavg = reduce(lambda x,y: x+y, (i[1] for i in solutions))/len(solutions)
    feromones = { k : (ro + th/Lavg)*v for (k,v) in feromones.items() }
    solutions.sort(key = lambda x: x[1])
    
    if(bestSolution is not None):
        if(solutions[0][1] < bestSolution[1]):
            bestSolution = solutions[0]
        for path in bestSolution[0]:
            for i in range(len(path)-1):
                feromones[(min(path[i],path[i+1]), max(path[i],path[i+1]))] = sigm/bestSolution[1] + feromones[(min(path[i],path[i+1]), max(path[i],path[i+1]))]
    else:
        bestSolution = solutions[0]
        
    for l in range(sigm):
        paths = solutions[l][0]
        L = solutions[l][1]
        for path in paths:
            for i in range(len(path)-1):
                feromones[(min(path[i],path[i+1]), max(path[i],path[i+1]))] = (sigm-(l+1)/L**(l+1)) + feromones[(min(path[i],path[i+1]), max(path[i],path[i+1]))]
    return bestSolution

def main():
    bestSolution = None
    vertices, edges, capacityLimit, demand, feromones, optimalValue = generateGraph()
    
    for i in range(iterations):
        solutions = list()
        for _ in range(ants):
            solution = solutionOfOneAnt(vertices.copy(), edges, capacityLimit, demand, feromones)
            solutions.append((solution, rateSolution(solution, edges)))
        bestSolution = updateFeromone(feromones, solutions, bestSolution)
        if i % 100 == 0:
            print(f"{i}:\t{int(bestSolution[1])}\t{optimalValue}")
        
    return bestSolution

if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "f:a:b:s:r:t:i:n:",["fileName=", "alpha=","beta=","sigma=","rho=","theta=","iterations=","numberOfAnts="])
    except getopt.GetoptError:
        print("""use: python aco_cvrp.py -f <fileName> -a <alpha> -b <beta> -s <sigma> -r <rho> -t <theta> -i <iterations> -n <numberOfAnts>""")
        sys.exit(2)
        
    for opt,arg in opts:
        if(opt in ("-a", "--alpha")): alfa = float(arg)
        elif(opt in ("-b", "--beta")): beta = float(arg)
        elif(opt in ("-s", "--sigma")): sigm = float(arg)
        elif(opt in ("-r", "--rho")): ro = float(arg)
        elif(opt in ("-t", "--theta")): th = float(arg)
        elif(opt in ("-f", "--fileName", "--file")): fileName = str(arg)
        elif(opt in ("-i", "--iterations")): iterations = int(arg)
        elif(opt in ("-n", "--numberOfAnts")): ants = int(arg)
        
    print(f"file name:\t{fileName}\nalpha:\t{alfa}\nbeta:\t{beta}\nsigma:\t{sigm}\nrho:\t{ro}\ntheta:\t{th}\niterations:\t{iterations}\nnumber of ants:\t{ants}")
    solution = main()
    print(f"Solution: {solution}")
    print(f"Total Distance: {solution[1]:.2f}")