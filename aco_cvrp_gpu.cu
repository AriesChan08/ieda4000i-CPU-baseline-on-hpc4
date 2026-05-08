
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// CUDA Error Check Macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Constants (default values, can be overridden)
__constant__ double d_alpha = 2.0;
__constant__ double d_beta = 5.0;
__constant__ double d_sigma = 3.0;
__constant__ double d_rho = 0.8;
__constant__ double d_theta = 80.0;
__constant__ int d_num_ants = 22;
__constant__ int d_num_vertices = 0;
__constant__ int d_capacity = 0;

// Structure to hold problem data
typedef struct {
    int num_vertices;
    int capacity;
    double* distances;  // num_vertices x num_vertices
    int* demands;       // num_vertices
    int* vertices;      // vertex indices (excluding depot)
    int num_customers;
} ProblemData;

// Structure for solution
typedef struct {
    int* paths;         // flattened paths
    int* path_lengths;  // length of each path
    int num_paths;
    double total_distance;
} Solution;

// Host-side parameters
double h_alpha = 2.0;
double h_beta = 5.0;
double h_sigma = 3.0;
double h_rho = 0.8;
double h_theta = 80.0;
int h_iterations = 1000;
int h_num_ants = 22;
char h_fileName[256] = "E-n22-k4.txt";

// Function to read CVRPLib file
ProblemData read_cvrplib(const char* file_path) {
    ProblemData data = {0};
    FILE* fp = fopen(file_path, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open file: %s\n", file_path);
        exit(EXIT_FAILURE);
    }
    
    char line[1024];
    int num_nodes = 0;
    double* x_coords = NULL;
    double* y_coords = NULL;
    int* demands = NULL;
    int capacity = 0;
    int reading_coords = 0;
    int reading_demands = 0;
    int reading_depot = 0;
    int node_count = 0;
    int demand_count = 0;
    
    // First pass: count nodes
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "DIMENSION")) {
            sscanf(line, "%*s %*s %d", &num_nodes);
        }
        if (strstr(line, "CAPACITY")) {
            sscanf(line, "%*s %*s %d", &capacity);
        }
    }
    
    if (num_nodes == 0) {
        fprintf(stderr, "Could not find DIMENSION in file\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    
    // Allocate memory
    x_coords = (double*)malloc(num_nodes * sizeof(double));
    y_coords = (double*)malloc(num_nodes * sizeof(double));
    demands = (int*)malloc(num_nodes * sizeof(int));
    
    // Second pass: read data
    rewind(fp);
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "NODE_COORD_SECTION")) {
            reading_coords = 1;
            reading_demands = 0;
            reading_depot = 0;
            continue;
        }
        if (strstr(line, "DEMAND_SECTION")) {
            reading_coords = 0;
            reading_demands = 1;
            reading_depot = 0;
            continue;
        }
        if (strstr(line, "DEPOT_SECTION")) {
            reading_coords = 0;
            reading_demands = 0;
            reading_depot = 1;
            continue;
        }
        if (strstr(line, "EOF")) break;
        
        if (reading_coords) {
            int id;
            double x, y;
            if (sscanf(line, "%d %lf %lf", &id, &x, &y) == 3) {
                if (id >= 1 && id <= num_nodes) {
                    x_coords[id-1] = x;
                    y_coords[id-1] = y;
                    node_count++;
                }
            }
        }
        if (reading_demands) {
            int id, d;
            if (sscanf(line, "%d %d", &id, &d) == 2) {
                if (id >= 1 && id <= num_nodes) {
                    demands[id-1] = d;
                    demand_count++;
                }
            }
        }
    }
    fclose(fp);
    
    // Setup problem data
    data.num_vertices = num_nodes;
    data.capacity = capacity;
    data.num_customers = num_nodes - 1;  // excluding depot (node 1)
    
    // Allocate distance matrix
    data.distances = (double*)malloc(num_nodes * num_nodes * sizeof(double));
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            if (i == j) {
                data.distances[i * num_nodes + j] = 0.0;
            } else {
                double dx = x_coords[i] - x_coords[j];
                double dy = y_coords[i] - y_coords[j];
                data.distances[i * num_nodes + j] = sqrt(dx*dx + dy*dy);
            }
        }
    }
    
    // Allocate demands
    data.demands = (int*)malloc(num_nodes * sizeof(int));
    memcpy(data.demands, demands, num_nodes * sizeof(int));
    
    // Allocate vertices (excluding depot)
    data.vertices = (int*)malloc(data.num_customers * sizeof(int));
    for (int i = 0; i < data.num_customers; i++) {
        data.vertices[i] = i + 2;  // vertices are 1-indexed, depot is 1
    }
    
    free(x_coords);
    free(y_coords);
    free(demands);
    
    return data;
}

// CUDA kernel for ant solution construction
__global__ void construct_solutions_kernel(
    double* d_distances,
    int* d_demands,
    int* d_vertices,
    int num_customers,
    int capacity,
    double* d_pheromones,
    double* d_solution_distances,
    int* d_solution_paths,
    int* d_solution_path_lengths,
    int* d_solution_num_paths,
    unsigned long long* seed_states
) {
    int ant_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant_id >= d_num_ants) return;
    
    // Initialize random state for this ant
    curandState state;
    curand_init(seed_states[ant_id], 0, 0, &state);
    
    // Shared memory for frequently accessed data
    __shared__ double shared_distances[1024];  // Adjust size as needed
    __shared__ int shared_demands[512];
    __shared__ double shared_pheromones[1024];
    
    // Copy data to shared memory (simplified - in practice would need more sophisticated caching)
    if (threadIdx.x < num_customers * num_customers && threadIdx.x < 1024) {
        shared_distances[threadIdx.x] = d_distances[threadIdx.x];
    }
    if (threadIdx.x < num_customers && threadIdx.x < 512) {
        shared_demands[threadIdx.x] = d_demands[threadIdx.x + 1];  // +1 because depot is index 0
    }
    __syncthreads();
    
    // Local arrays for this ant
    int local_vertices[512];  // Max customers
    int local_visited[512] = {0};
    int local_paths[1024];    // Max path storage
    int local_path_lengths[512];
    int local_num_paths = 0;
    int path_start = 0;
    
    // Copy vertices to local memory
    int num_remaining = num_customers;
    for (int i = 0; i < num_customers; i++) {
        local_vertices[i] = d_vertices[i];
    }
    
    double total_distance = 0.0;
    
    while (num_remaining > 0) {
        // Select first city randomly
        int idx = (int)(curand_uniform(&state) * num_remaining);
        int current_city = local_vertices[idx];
        
        // Remove selected city
        local_vertices[idx] = local_vertices[num_remaining - 1];
        num_remaining--;
        
        int current_capacity = capacity - d_demands[current_city - 1];
        local_paths[path_start] = current_city;
        int path_len = 1;
        
        while (num_remaining > 0) {
            // Calculate probabilities
            double prob_sum = 0.0;
            double probs[512];
            
            for (int i = 0; i < num_remaining; i++) {
                int next_city = local_vertices[i];
                int min_city = min(current_city, next_city);
                int max_city = max(current_city, next_city);
                
                // Access pheromone (simplified - would need proper indexing)
                double tau = d_pheromones[min_city * d_num_vertices + max_city];
                double eta = 1.0 / d_distances[(current_city-1) * d_num_vertices + (next_city-1)];
                
                probs[i] = pow(tau, d_alpha) * pow(eta, d_beta);
                prob_sum += probs[i];
            }
            
            if (prob_sum < 1e-10) break;
            
            // Normalize probabilities
            for (int i = 0; i < num_remaining; i++) {
                probs[i] /= prob_sum;
            }
            
            // Select next city using roulette wheel
            double r = curand_uniform(&state);
            double cum_prob = 0.0;
            int selected_idx = -1;
            
            for (int i = 0; i < num_remaining; i++) {
                cum_prob += probs[i];
                if (r <= cum_prob) {
                    selected_idx = i;
                    break;
                }
            }
            
            if (selected_idx == -1) selected_idx = num_remaining - 1;
            
            int next_city = local_vertices[selected_idx];
            
            // Check capacity constraint
            if (current_capacity - d_demands[next_city - 1] > 0) {
                current_capacity -= d_demands[next_city - 1];
                local_paths[path_start + path_len] = next_city;
                path_len++;
                
                // Add edge distance
                total_distance += d_distances[(current_city-1) * d_num_vertices + (next_city-1)];
                current_city = next_city;
                
                // Remove selected city
                local_vertices[selected_idx] = local_vertices[num_remaining - 1];
                num_remaining--;
            } else {
                break;
            }
        }
        
        // Add return to depot
        total_distance += d_distances[(current_city-1) * d_num_vertices + 0];  // Return to depot (index 0)
        
        // Store path
        local_path_lengths[local_num_paths] = path_len;
        local_num_paths++;
        path_start += path_len;
    }
    
    // Write results to global memory
    d_solution_distances[ant_id] = total_distance;
    d_solution_num_paths[ant_id] = local_num_paths;
    
    for (int i = 0; i < path_start; i++) {
        d_solution_paths[ant_id * 1024 + i] = local_paths[i];
    }
    for (int i = 0; i < local_num_paths; i++) {
        d_solution_path_lengths[ant_id * 512 + i] = local_path_lengths[i];
    }
    
    // Update seed for next iteration
    seed_states[ant_id] = curand(&state);
}

// CUDA kernel for pheromone update
__global__ void update_pheromones_kernel(
    double* d_pheromones,
    double* d_solution_distances,
    int* d_solution_paths,
    int* d_solution_path_lengths,
    int* d_solution_num_paths,
    double best_distance,
    int* d_best_paths,
    int* d_best_path_lengths,
    int best_num_paths
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_edges = d_num_vertices * d_num_vertices;
    
    if (idx >= total_edges) return;
    
    int i = idx / d_num_vertices;
    int j = idx % d_num_vertices;
    
    if (i >= j) return;  // Only process upper triangle
    
    // Evaporation
    double Lavg = 0.0;
    for (int a = 0; a < d_num_ants; a++) {
        Lavg += d_solution_distances[a];
    }
    Lavg /= d_num_ants;
    
    double new_pheromone = (d_rho + d_theta / Lavg) * d_pheromones[idx];
    
    // Add pheromone from best solution
    if (best_distance > 0) {
        // Check if edge is in best solution
        int path_offset = 0;
        for (int p = 0; p < best_num_paths; p++) {
            int path_len = d_best_path_lengths[p];
            for (int k = 0; k < path_len - 1; k++) {
                int city1 = d_best_paths[path_offset + k];
                int city2 = d_best_paths[path_offset + k + 1];
                int min_city = min(city1, city2);
                int max_city = max(city1, city2);
                if (min_city == i && max_city == j) {
                    new_pheromone += d_sigma / best_distance;
                }
            }
            path_offset += path_len;
        }
    }
    
    // Add pheromone from top sigma solutions
    for (int l = 0; l < (int)d_sigma && l < d_num_ants; l++) {
        double L = d_solution_distances[l];
        int path_offset = 0;
        for (int p = 0; p < d_solution_num_paths[l]; p++) {
            int path_len = d_solution_path_lengths[l * 512 + p];
            for (int k = 0; k < path_len - 1; k++) {
                int city1 = d_solution_paths[l * 1024 + path_offset + k];
                int city2 = d_solution_paths[l * 1024 + path_offset + k + 1];
                int min_city = min(city1, city2);
                int max_city = max(city1, city2);
                if (min_city == i && max_city == j) {
                    new_pheromone += (d_sigma - (l + 1)) / pow(L, l + 1);
                }
            }
            path_offset += path_len;
        }
    }
    
    d_pheromones[idx] = new_pheromone;
}

// Host function to run ACO algorithm
void run_aco(ProblemData& data) {
    int num_vertices = data.num_vertices;
    int num_customers = data.num_customers;
    int capacity = data.capacity;
    
    // Device memory allocations
    double *d_distances, *d_pheromones, *d_solution_distances;
    int *d_demands, *d_vertices, *d_solution_paths, *d_solution_path_lengths, *d_solution_num_paths;
    int *d_best_paths, *d_best_path_lengths;
    unsigned long long *d_seed_states;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_distances, num_vertices * num_vertices * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pheromones, num_vertices * num_vertices * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_demands, num_vertices * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vertices, num_customers * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_solution_distances, h_num_ants * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_solution_paths, h_num_ants * 1024 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_solution_path_lengths, h_num_ants * 512 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_solution_num_paths, h_num_ants * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_paths, h_num_ants * 1024 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_path_lengths, h_num_ants * 512 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seed_states, h_num_ants * sizeof(unsigned long long)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_distances, data.distances, num_vertices * num_vertices * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_demands, data.demands, num_vertices * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vertices, data.vertices, num_customers * sizeof(int), cudaMemcpyHostToDevice));
    
    // Initialize pheromones to 1.0
    double* h_pheromones = (double*)malloc(num_vertices * num_vertices * sizeof(double));
    for (int i = 0; i < num_vertices * num_vertices; i++) {
        h_pheromones[i] = 1.0;
    }
    CUDA_CHECK(cudaMemcpy(d_pheromones, h_pheromones, num_vertices * num_vertices * sizeof(double), cudaMemcpyHostToDevice));
    
    // Initialize random seeds
    unsigned long long* h_seed_states = (unsigned long long*)malloc(h_num_ants * sizeof(unsigned long long));
    srand(time(NULL));
    for (int i = 0; i < h_num_ants; i++) {
        h_seed_states[i] = (unsigned long long)rand() * (unsigned long long)rand();
    }
    CUDA_CHECK(cudaMemcpy(d_seed_states, h_seed_states, h_num_ants * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    
    // Set constant memory
    int num_vertices_const = num_vertices;
    int capacity_const = capacity;
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_vertices, &num_vertices_const, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_capacity, &capacity_const, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_ants, &h_num_ants, sizeof(int)));
    
    // Host arrays for results
    double* h_solution_distances = (double*)malloc(h_num_ants * sizeof(double));
    int* h_solution_paths = (int*)malloc(h_num_ants * 1024 * sizeof(int));
    int* h_solution_path_lengths = (int*)malloc(h_num_ants * 512 * sizeof(int));
    int* h_solution_num_paths = (int*)malloc(h_num_ants * sizeof(int));
    int* h_best_paths = (int*)malloc(h_num_ants * 1024 * sizeof(int));
    int* h_best_path_lengths = (int*)malloc(h_num_ants * 512 * sizeof(int));
    
    double best_distance = DBL_MAX;
    int best_num_paths = 0;
    
    // CUDA grid configuration
    int block_size = 256;
    int num_blocks = (h_num_ants + block_size - 1) / block_size;
    int pheromone_blocks = (num_vertices * num_vertices + block_size - 1) / block_size;
    
    // Main iteration loop
    for (int iter = 0; iter < h_iterations; iter++) {
        // Launch ant construction kernel
        construct_solutions_kernel<<<num_blocks, block_size>>>(
            d_distances, d_demands, d_vertices, num_customers, capacity,
            d_pheromones, d_solution_distances, d_solution_paths,
            d_solution_path_lengths, d_solution_num_paths, d_seed_states
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy results back to host
        CUDA_CHECK(cudaMemcpy(h_solution_distances, d_solution_distances, h_num_ants * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_solution_paths, d_solution_paths, h_num_ants * 1024 * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_solution_path_lengths, d_solution_path_lengths, h_num_ants * 512 * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_solution_num_paths, d_solution_num_paths, h_num_ants * sizeof(int), cudaMemcpyDeviceToHost));
        
        // Find best solution in this iteration
        double iter_best = DBL_MAX;
        int best_idx = 0;
        for (int i = 0; i < h_num_ants; i++) {
            if (h_solution_distances[i] < iter_best) {
                iter_best = h_solution_distances[i];
                best_idx = i;
            }
        }
        
        // Update global best
        if (iter_best < best_distance) {
            best_distance = iter_best;
            best_num_paths = h_solution_num_paths[best_idx];
            
            // Copy best solution
            memcpy(h_best_paths, &h_solution_paths[best_idx * 1024], 1024 * sizeof(int));
            memcpy(h_best_path_lengths, &h_solution_path_lengths[best_idx * 512], 512 * sizeof(int));
            
            // Copy to device
            CUDA_CHECK(cudaMemcpy(d_best_paths, h_best_paths, h_num_ants * 1024 * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_best_path_lengths, h_best_path_lengths, h_num_ants * 512 * sizeof(int), cudaMemcpyHostToDevice));
        }
        
        // Launch pheromone update kernel
        update_pheromones_kernel<<<pheromone_blocks, block_size>>>(
            d_pheromones, d_solution_distances, d_solution_paths,
            d_solution_path_lengths, d_solution_num_paths,
            best_distance, d_best_paths, d_best_path_lengths, best_num_paths
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        if (iter % 100 == 0) {
            printf("%d:\t%.0f\n", iter, best_distance);
        }
    }
    
    printf("Total Distance: %.1f\n", best_distance);
    
    // Cleanup
    free(h_pheromones);
    free(h_seed_states);
    free(h_solution_distances);
    free(h_solution_paths);
    free(h_solution_path_lengths);
    free(h_solution_num_paths);
    free(h_best_paths);
    free(h_best_path_lengths);
    
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_pheromones));
    CUDA_CHECK(cudaFree(d_demands));
    CUDA_CHECK(cudaFree(d_vertices));
    CUDA_CHECK(cudaFree(d_solution_distances));
    CUDA_CHECK(cudaFree(d_solution_paths));
    CUDA_CHECK(cudaFree(d_solution_path_lengths));
    CUDA_CHECK(cudaFree(d_solution_num_paths));
    CUDA_CHECK(cudaFree(d_best_paths));
    CUDA_CHECK(cudaFree(d_best_path_lengths));
    CUDA_CHECK(cudaFree(d_seed_states));
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            strcpy(h_fileName, argv[++i]);
        } else if (strcmp(argv[i], "-a") == 0 && i + 1 < argc) {
            h_alpha = atof(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            h_beta = atof(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            h_sigma = atof(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            h_rho = atof(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            h_theta = atof(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            h_iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            h_num_ants = atoi(argv[++i]);
        }
    }
    
    printf("file name:\t%s\n", h_fileName);
    printf("alpha:\t%.1f\n", h_alpha);
    printf("beta:\t%.1f\n", h_beta);
    printf("sigma:\t%.1f\n", h_sigma);
    printf("rho:\t%.1f\n", h_rho);
    printf("theta:\t%.1f\n", h_theta);
    printf("iterations:\t%d\n", h_iterations);
    printf("number of ants:\t%d\n", h_num_ants);
    
    // Read problem data
    ProblemData data = read_cvrplib(h_fileName);
    
    // Run ACO algorithm
    run_aco(data);
    
    // Cleanup
    free(data.distances);
    free(data.demands);
    free(data.vertices);
    
    return 0;
}
