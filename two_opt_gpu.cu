
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 数据结构
struct Node {
    int id;
    float x, y;
    int demand;
};

// 读取CVRPLib文件
void readCVRPLib(const std::string& filePath, std::vector<Node>& nodes, int& capacity) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filePath << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    file.close();

    // 查找各段起始位置
    int nodeCoordStart = -1, demandStart = -1, depotStart = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (lines[i].find("NODE_COORD_SECTION") != std::string::npos) nodeCoordStart = i + 1;
        if (lines[i].find("DEMAND_SECTION") != std::string::npos) demandStart = i + 1;
        if (lines[i].find("DEPOT_SECTION") != std::string::npos) depotStart = i + 1;
        if (lines[i].find("CAPACITY") != std::string::npos) {
            std::stringstream ss(lines[i]);
            std::string token;
            while (ss >> token) {
                if (std::isdigit(token[0])) {
                    capacity = std::stoi(token);
                    break;
                }
            }
        }
    }

    if (nodeCoordStart == -1 || demandStart == -1 || depotStart == -1) {
        std::cerr << "Error: Invalid CVRPLib file format" << std::endl;
        exit(EXIT_FAILURE);
    }

    // 读取坐标
    std::vector<std::pair<int, std::pair<float, float>>> tempCoords;
    for (int i = nodeCoordStart; i < demandStart - 1 && i < (int)lines.size(); ++i) {
        std::stringstream ss(lines[i]);
        int id;
        float x, y;
        if (ss >> id >> x >> y) {
            tempCoords.push_back({id, {x, y}});
        }
    }

    // 读取需求
    std::vector<std::pair<int, int>> tempDemands;
    for (int i = demandStart; i < depotStart - 1 && i < (int)lines.size(); ++i) {
        std::stringstream ss(lines[i]);
        int id, d;
        if (ss >> id >> d) {
            tempDemands.push_back({id, d});
        }
    }

    // 合并数据
    nodes.clear();
    for (const auto& coord : tempCoords) {
        Node node;
        node.id = coord.first;
        node.x = coord.second.first;
        node.y = coord.second.second;
        node.demand = 0;
        for (const auto& dem : tempDemands) {
            if (dem.first == node.id) {
                node.demand = dem.second;
                break;
            }
        }
        nodes.push_back(node);
    }

    // 按ID排序
    std::sort(nodes.begin(), nodes.end(), [](const Node& a, const Node& b) {
        return a.id < b.id;
    });
}

// 计算两点间距离
__host__ __device__ float calculateDistance(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return sqrtf(dx * dx + dy * dy);
}

// 创建距离矩阵（主机端）
void createDistanceMatrix(const std::vector<Node>& nodes, std::vector<float>& distMatrix, int& n) {
    n = nodes.size();
    distMatrix.resize(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            distMatrix[i * n + j] = calculateDistance(nodes[i].x, nodes[i].y, nodes[j].x, nodes[j].y);
        }
    }
}

// 计算路径距离（主机端）
float calculateRouteDistance(const std::vector<int>& route, const std::vector<float>& distMatrix, int n) {
    float total = 0.0f;
    for (size_t i = 0; i < route.size() - 1; ++i) {
        total += distMatrix[route[i] * n + route[i + 1]];
    }
    return total;
}

// CUDA核函数：计算交换后的路径距离
__global__ void twoOptKernel(const float* distMatrix, const int* route, int* bestI, int* bestK, 
                              float* bestDistance, int routeLen, int n, bool* improved) {
    __shared__ float sharedDist[256]; // 共享内存用于部分结果
    __shared__ int sharedBestI[1];
    __shared__ int sharedBestK[1];
    __shared__ float sharedBestDist[1];
    __shared__ bool sharedImproved[1];

    int tid = threadIdx.x;
    int totalThreads = blockDim.x;

    // 初始化共享内存
    if (tid == 0) {
        sharedBestI[0] = -1;
        sharedBestK[0] = -1;
        sharedBestDist[0] = *bestDistance;
        sharedImproved[0] = false;
    }
    __syncthreads();

    // 计算需要检查的(i,k)对数量
    int numPairs = (routeLen - 3) * (routeLen - 2) / 2;
    int pairsPerThread = (numPairs + totalThreads - 1) / totalThreads;
    int startPair = tid * pairsPerThread;
    int endPair = min(startPair + pairsPerThread, numPairs);

    // 将(i,k)对映射到线性索引
    for (int pairIdx = startPair; pairIdx < endPair; ++pairIdx) {
        // 将线性索引映射回(i,k)
        int i = 1;
        int k = i + 1;
        int count = 0;
        while (count < pairIdx) {
            if (k < routeLen - 1) {
                k++;
            } else {
                i++;
                k = i + 1;
            }
            count++;
        }

        // 计算交换后的距离
        float oldDist = 0.0f;
        float newDist = 0.0f;

        // 移除旧边
        oldDist += distMatrix[route[i-1] * n + route[i]];
        oldDist += distMatrix[route[k] * n + route[k+1]];

        // 添加新边
        newDist += distMatrix[route[i-1] * n + route[k]];
        newDist += distMatrix[route[i] * n + route[k+1]];

        // 对于中间部分，距离不变
        float delta = newDist - oldDist;

        if (delta < 0) {
            float newTotalDist = *bestDistance + delta;
            if (newTotalDist < sharedBestDist[0]) {
                sharedBestDist[0] = newTotalDist;
                sharedBestI[0] = i;
                sharedBestK[0] = k;
                sharedImproved[0] = true;
            }
        }
    }

    __syncthreads();

    // 将结果写回全局内存
    if (tid == 0) {
        if (sharedImproved[0]) {
            *bestI = sharedBestI[0];
            *bestK = sharedBestK[0];
            *bestDistance = sharedBestDist[0];
            *improved = true;
        }
    }
}

// CUDA核函数：执行2-opt交换
__global__ void swapKernel(int* route, int i, int k, int routeLen) {
    int tid = threadIdx.x;
    int totalThreads = blockDim.x;

    // 反转route[i]到route[k]之间的部分
    int segmentLen = k - i + 1;
    int halfLen = segmentLen / 2;

    for (int idx = tid; idx < halfLen; idx += totalThreads) {
        int left = i + idx;
        int right = k - idx;
        int temp = route[left];
        route[left] = route[right];
        route[right] = temp;
    }
}

// 在GPU上执行2-opt优化
void twoOptGPU(std::vector<int>& route, const std::vector<float>& distMatrix, int n) {
    int routeLen = route.size();
    if (routeLen <= 3) return; // 不需要优化

    // 分配设备内存
    float* d_distMatrix;
    int* d_route;
    int* d_bestI;
    int* d_bestK;
    float* d_bestDistance;
    bool* d_improved;

    CUDA_CHECK(cudaMalloc(&d_distMatrix, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_route, routeLen * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bestI, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bestK, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bestDistance, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_improved, sizeof(bool)));

    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_distMatrix, distMatrix.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_route, route.data(), routeLen * sizeof(int), cudaMemcpyHostToDevice));

    float currentDist = calculateRouteDistance(route, distMatrix, n);
    CUDA_CHECK(cudaMemcpy(d_bestDistance, &currentDist, sizeof(float), cudaMemcpyHostToDevice));

    bool improved = true;
    int maxIterations = 1000; // 防止无限循环
    int iter = 0;

    while (improved && iter < maxIterations) {
        improved = false;
        CUDA_CHECK(cudaMemcpy(d_improved, &improved, sizeof(bool), cudaMemcpyHostToDevice));

        // 启动核函数
        int blockSize = 256;
        int gridSize = 1; // 使用单个block，因为共享内存限制
        twoOptKernel<<<gridSize, blockSize>>>(d_distMatrix, d_route, d_bestI, d_bestK, 
                                               d_bestDistance, routeLen, n, d_improved);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 检查是否找到改进
        CUDA_CHECK(cudaMemcpy(&improved, d_improved, sizeof(bool), cudaMemcpyDeviceToHost));

        if (improved) {
            int bestI, bestK;
            float newDist;
            CUDA_CHECK(cudaMemcpy(&bestI, d_bestI, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&bestK, d_bestK, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&newDist, d_bestDistance, sizeof(float), cudaMemcpyDeviceToHost));

            // 执行交换
            swapKernel<<<1, 256>>>(d_route, bestI, bestK, routeLen);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            currentDist = newDist;
        }
        iter++;
    }

    // 将结果拷贝回主机
    CUDA_CHECK(cudaMemcpy(route.data(), d_route, routeLen * sizeof(int), cudaMemcpyDeviceToHost));

    // 释放设备内存
    CUDA_CHECK(cudaFree(d_distMatrix));
    CUDA_CHECK(cudaFree(d_route));
    CUDA_CHECK(cudaFree(d_bestI));
    CUDA_CHECK(cudaFree(d_bestK));
    CUDA_CHECK(cudaFree(d_bestDistance));
    CUDA_CHECK(cudaFree(d_improved));
}

// 求解CVRP
void solveCVRP(const std::vector<Node>& nodes, int capacity, std::vector<std::vector<int>>& finalRoutes, float& totalDistance) {
    int n = nodes.size();
    std::vector<float> distMatrix;
    createDistanceMatrix(nodes, distMatrix, n);

    // 找到depot（ID为1的节点）
    int depotIdx = -1;
    for (int i = 0; i < n; ++i) {
        if (nodes[i].id == 1) {
            depotIdx = i;
            break;
        }
    }

    if (depotIdx == -1) {
        std::cerr << "Error: Depot (node 1) not found" << std::endl;
        exit(EXIT_FAILURE);
    }

    // 构建初始路径（贪心算法）
    std::vector<int> customers;
    for (int i = 0; i < n; ++i) {
        if (i != depotIdx) {
            customers.push_back(i);
        }
    }

    std::vector<std::vector<int>> routes;
    std::vector<int> currentRoute = {depotIdx};
    int currentLoad = 0;

    for (int custIdx : customers) {
        int custDemand = nodes[custIdx].demand;
        if (currentLoad + custDemand <= capacity) {
            currentRoute.push_back(custIdx);
            currentLoad += custDemand;
        } else {
            currentRoute.push_back(depotIdx);
            routes.push_back(currentRoute);
            currentRoute = {depotIdx, custIdx};
            currentLoad = custDemand;
        }
    }
    currentRoute.push_back(depotIdx);
    routes.push_back(currentRoute);

    // 对每条路径进行2-opt优化
    totalDistance = 0.0f;
    finalRoutes.clear();
    for (auto& route : routes) {
        twoOptGPU(route, distMatrix, n);
        finalRoutes.push_back(route);
        totalDistance += calculateRouteDistance(route, distMatrix, n);
    }

    // 将索引转换回节点ID
    for (auto& route : finalRoutes) {
        for (auto& idx : route) {
            idx = nodes[idx].id;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <CVRPLib file path>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string filePath = argv[1];
    std::vector<Node> nodes;
    int capacity = 0;

    readCVRPLib(filePath, nodes, capacity);

    std::vector<std::vector<int>> routes;
    float totalDistance;

    solveCVRP(nodes, capacity, routes, totalDistance);

    // 输出结果
    std::cout << "Total Distance: " << totalDistance << std::endl;
    for (size_t i = 0; i < routes.size(); ++i) {
        std::cout << "Route " << i + 1 << ": ";
        for (size_t j = 0; j < routes[i].size(); ++j) {
            std::cout << routes[i][j];
            if (j < routes[i].size() - 1) std::cout << " -> ";
        }
        std::cout << std::endl;
    }

    return 0;
}
