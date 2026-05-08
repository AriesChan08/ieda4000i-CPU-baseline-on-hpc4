
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ==================== CUDA Error Checking Macro ====================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ==================== Data Structures ====================
struct Coord {
    double x, y;
};

struct Saving {
    float saving;
    int i, j;
};

// ==================== Host Functions ====================
void readCVRPLib(const std::string& filePath, std::vector<Coord>& coords,
                 std::vector<int>& demand, int& capacity) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filePath << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    std::unordered_map<int, Coord> coordMap;
    std::unordered_map<int, int> demandMap;
    int depotNode = -1;
    bool inCoordSection = false, inDemandSection = false, inDepotSection = false;

    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line.find("NODE_COORD_SECTION") != std::string::npos) {
            inCoordSection = true;
            inDemandSection = false;
            inDepotSection = false;
            continue;
        }
        if (line.find("DEMAND_SECTION") != std::string::npos) {
            inCoordSection = false;
            inDemandSection = true;
            inDepotSection = false;
            continue;
        }
        if (line.find("DEPOT_SECTION") != std::string::npos) {
            inCoordSection = false;
            inDemandSection = false;
            inDepotSection = true;
            continue;
        }
        if (line.find("EOF") != std::string::npos) break;

        if (line.find("CAPACITY") != std::string::npos) {
            std::stringstream ss(line);
            std::string token;
            while (ss >> token) {
                if (std::isdigit(token[0])) {
                    capacity = std::stoi(token);
                    break;
                }
            }
            continue;
        }

        if (inCoordSection) {
            std::stringstream ss(line);
            int id;
            double x, y;
            if (ss >> id >> x >> y) {
                coordMap[id] = {x, y};
            }
        } else if (inDemandSection) {
            std::stringstream ss(line);
            int id, d;
            if (ss >> id >> d) {
                demandMap[id] = d;
            }
        } else if (inDepotSection) {
            std::stringstream ss(line);
            int id;
            if (ss >> id && id != -1) {
                depotNode = id;
            }
        }
    }
    file.close();

    // Build sorted node list (depot first)
    std::vector<int> nodes;
    nodes.push_back(depotNode);
    for (auto& p : coordMap) {
        if (p.first != depotNode) nodes.push_back(p.first);
    }
    std::sort(nodes.begin() + 1, nodes.end());

    // Fill output arrays
    coords.resize(nodes.size());
    demand.resize(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        coords[i] = coordMap[nodes[i]];
        demand[i] = demandMap[nodes[i]];
    }
}

// ==================== CUDA Kernels ====================
__global__ void computeDistancesKernel(const Coord* coords, float* distMatrix, int n) {
    __shared__ float sharedCoordsX[256], sharedCoordsY[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPairs = n * n;
    
    // Cooperative loading of coordinates into shared memory
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sharedCoordsX[i] = coords[i].x;
        sharedCoordsY[i] = coords[i].y;
    }
    __syncthreads();
    
    if (idx < totalPairs) {
        int i = idx / n;
        int j = idx % n;
        double dx = sharedCoordsX[i] - sharedCoordsX[j];
        double dy = sharedCoordsY[i] - sharedCoordsY[j];
        distMatrix[idx] = sqrtf(dx * dx + dy * dy);
    }
}

__global__ void computeSavingsKernel(const float* distMatrix, Saving* savings, 
                                     int* savingCount, int n, int numCustomers) {
    extern __shared__ float sharedDist[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Each block processes a chunk of customer pairs
    int customersPerBlock = (numCustomers + gridDim.x - 1) / gridDim.x;
    int startCustomer = bid * customersPerBlock;
    int endCustomer = min(startCustomer + customersPerBlock, numCustomers);
    
    // Load distance matrix rows for this block's customers into shared memory
    // We need distances from depot (index 0) and between customers
    for (int i = tid; i < n * n; i += blockDim.x) {
        sharedDist[i] = distMatrix[i];
    }
    __syncthreads();
    
    int localCount = 0;
    Saving localSavings[256]; // Max per thread
    
    for (int i = startCustomer + tid; i < endCustomer; i += blockDim.x) {
        int ci = i + 1; // customer index (0-based, depot is 0)
        for (int j = i + 1; j < numCustomers; ++j) {
            int cj = j + 1;
            float s = sharedDist[0 * n + ci] + sharedDist[0 * n + cj] - sharedDist[ci * n + cj];
            if (localCount < 256) {
                localSavings[localCount++] = {-s, ci, cj};
            }
        }
    }
    
    // Write local savings to global memory atomically
    int base = atomicAdd(savingCount, localCount);
    for (int k = 0; k < localCount; ++k) {
        savings[base + k] = localSavings[k];
    }
}

// ==================== Host Algorithm ====================
std::vector<std::vector<int>> clarkeWrightSavings(
    const std::vector<Coord>& coords,
    const std::vector<int>& demand,
    int capacity) {
    
    int n = coords.size();
    int numCustomers = n - 1;
    
    // Allocate device memory
    Coord* d_coords;
    float* d_distMatrix;
    Saving* d_savings;
    int* d_savingCount;
    
    CUDA_CHECK(cudaMalloc(&d_coords, n * sizeof(Coord)));
    CUDA_CHECK(cudaMalloc(&d_distMatrix, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_savings, numCustomers * numCustomers * sizeof(Saving)));
    CUDA_CHECK(cudaMalloc(&d_savingCount, sizeof(int)));
    
    // Copy coordinates to device
    CUDA_CHECK(cudaMemcpy(d_coords, coords.data(), n * sizeof(Coord), cudaMemcpyHostToDevice));
    
    // Launch distance computation kernel
    int totalPairs = n * n;
    int threadsPerBlock = 256;
    int blocks = (totalPairs + threadsPerBlock - 1) / threadsPerBlock;
    computeDistancesKernel<<<blocks, threadsPerBlock>>>(d_coords, d_distMatrix, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Launch savings computation kernel
    CUDA_CHECK(cudaMemset(d_savingCount, 0, sizeof(int)));
    int savingsBlocks = min(256, numCustomers);
    int sharedMemSize = n * n * sizeof(float);
    computeSavingsKernel<<<savingsBlocks, threadsPerBlock, sharedMemSize>>>(
        d_distMatrix, d_savings, d_savingCount, n, numCustomers);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy savings back to host
    int h_savingCount;
    CUDA_CHECK(cudaMemcpy(&h_savingCount, d_savingCount, sizeof(int), cudaMemcpyDeviceToHost));
    
    std::vector<Saving> h_savings(h_savingCount);
    CUDA_CHECK(cudaMemcpy(h_savings.data(), d_savings, h_savingCount * sizeof(Saving), 
                          cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_coords));
    CUDA_CHECK(cudaFree(d_distMatrix));
    CUDA_CHECK(cudaFree(d_savings));
    CUDA_CHECK(cudaFree(d_savingCount));
    
    // Sort savings on host
    std::sort(h_savings.begin(), h_savings.end(), 
              [](const Saving& a, const Saving& b) { return a.saving < b.saving; });
    
    // Build initial routes
    std::vector<std::vector<int>> routes(numCustomers);
    for (int i = 0; i < numCustomers; ++i) {
        routes[i] = {0, i + 1, 0}; // depot, customer, depot
    }
    
    // Merge routes based on savings
    for (const auto& s : h_savings) {
        int i = s.i, j = s.j;
        
        // Find routes containing i and j
        int route_i_idx = -1, route_j_idx = -1;
        for (int r = 0; r < (int)routes.size(); ++r) {
            if (std::find(routes[r].begin(), routes[r].end(), i) != routes[r].end()) {
                route_i_idx = r;
            }
            if (std::find(routes[r].begin(), routes[r].end(), j) != routes[r].end()) {
                route_j_idx = r;
            }
        }
        
        if (route_i_idx == -1 || route_j_idx == -1 || route_i_idx == route_j_idx) continue;
        
        auto& route_i = routes[route_i_idx];
        auto& route_j = routes[route_j_idx];
        
        // Check if i is at end of route_i and j is at start of route_j
        if (route_i[route_i.size() - 2] == i && route_j[1] == j) {
            // Calculate total demand
            int totalDemand = 0;
            for (size_t k = 1; k < route_i.size() - 1; ++k) totalDemand += demand[route_i[k]];
            for (size_t k = 1; k < route_j.size() - 1; ++k) totalDemand += demand[route_j[k]];
            
            if (totalDemand <= capacity) {
                // Merge routes
                std::vector<int> newRoute(route_i.begin(), route_i.end() - 1);
                newRoute.insert(newRoute.end(), route_j.begin() + 1, route_j.end());
                
                // Update all customers in new route to point to this route
                for (size_t k = 1; k < newRoute.size() - 1; ++k) {
                    int cust = newRoute[k];
                    for (auto& r : routes) {
                        if (std::find(r.begin(), r.end(), cust) != r.end()) {
                            r = newRoute;
                            break;
                        }
                    }
                }
            }
        }
    }
    
    // Remove duplicate routes
    std::set<std::vector<int>> uniqueRoutes;
    for (const auto& route : routes) {
        uniqueRoutes.insert(route);
    }
    
    return std::vector<std::vector<int>>(uniqueRoutes.begin(), uniqueRoutes.end());
}

// ==================== Main Function ====================
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <CVRPLib file>" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::string filePath = argv[1];
    
    // Read input
    std::vector<Coord> coords;
    std::vector<int> demand;
    int capacity = 0;
    readCVRPLib(filePath, coords, demand, capacity);
    
    // Run algorithm
    auto routes = clarkeWrightSavings(coords, demand, capacity);
    
    // Calculate total distance
    double totalDist = 0.0;
    for (const auto& route : routes) {
        for (size_t i = 0; i < route.size() - 1; ++i) {
            int from = route[i], to = route[i + 1];
            double dx = coords[from].x - coords[to].x;
            double dy = coords[from].y - coords[to].y;
            totalDist += std::sqrt(dx * dx + dy * dy);
        }
    }
    
    // Output results
    std::cout.precision(1);
    std::cout << std::fixed;
    std::cout << "Total Distance: " << totalDist << std::endl;
    
    for (size_t r = 0; r < routes.size(); ++r) {
        std::cout << "Route " << (r + 1) << ": ";
        for (size_t i = 0; i < routes[r].size(); ++i) {
            if (i > 0) std::cout << " -> ";
            std::cout << routes[r][i];
        }
        std::cout << std::endl;
    }
    
    return EXIT_SUCCESS;
}
