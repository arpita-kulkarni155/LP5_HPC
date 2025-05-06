#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

class Graph {
public:
    int vertices;
    vector<vector<int>> graph;
    vector<bool> visited;

    Graph(int v) : vertices(v) {
        graph.assign(vertices, vector<int>());
        visited.assign(vertices, false);
    }

    void addEdge(int a, int b) {
        graph[a].push_back(b);
        graph[b].push_back(a);
    }

    // Sequential DFS
    void dfs(int start) {
        stack<int> s;
        s.push(start);
        visited[start] = true;

        while (!s.empty()) {
            int current = s.top();
            cout << current << " ";
            s.pop();

            for (int neighbor : graph[current]) {
                if (!visited[neighbor]) {
                    s.push(neighbor);
                    visited[neighbor] = true;
                }
            }
        }
    }

    // Parallel DFS with Task-based Parallelism
    void parallel_dfs(int start) {
        visited.assign(vertices, false);  // Reset visited array for parallel DFS
        #pragma omp parallel
        {
            #pragma omp single
            dfs_task(start);
        }
    }

    void dfs_task(int start) {
        stack<int> s;
        s.push(start);
        visited[start] = true;

        while (!s.empty()) {
            int current = s.top();
            s.pop();
            cout << current << " ";

            // Create tasks for unvisited neighbors
            #pragma omp parallel for
            for (int i = 0; i < graph[current].size(); i++) {
                int neighbor = graph[current][i];
                if (!visited[neighbor]) {
                    #pragma omp critical
                    {
                        s.push(neighbor);
                        visited[neighbor] = true;
                    }
                }
            }
        }
    }

    // Sequential BFS
    void bfs(int start) {
        queue<int> q;
        q.push(start);
        visited[start] = true;

        while (!q.empty()) {
            int current = q.front();
            cout << current << " ";
            q.pop();

            for (int neighbor : graph[current]) {
                if (!visited[neighbor]) {
                    q.push(neighbor);
                    visited[neighbor] = true;
                }
            }
        }
    }

    // Parallel BFS with Task-based Parallelism
    void parallel_bfs(int start) {
        visited.assign(vertices, false);  // Reset visited array for parallel BFS
        #pragma omp parallel
        {
            #pragma omp single
            bfs_task(start);
        }
    }

    void bfs_task(int start) {
        queue<int> q;
        q.push(start);
        visited[start] = true;

        while (!q.empty()) {
            int current = q.front();
            q.pop();
            cout << current << " ";

            // Parallelize the exploration of neighbors
            #pragma omp parallel for
            for (int i = 0; i < graph[current].size(); i++) {
                int neighbor = graph[current][i];
                if (!visited[neighbor]) {
                    #pragma omp critical
                    {
                        q.push(neighbor);
                        visited[neighbor] = true;
                    }
                }
            }
        }
    }
};

int main() {
    int vertices = 100;  // Example with 1000 vertices
    int edges = 20000;  // Example with 5000 edges
    Graph g(vertices);

    // Generate random edges for the graph
    for (int i = 0; i < edges; i++) {
        int a = rand() % vertices;
        int b = rand() % vertices;
        if (a != b) {  // avoid self-loops
            g.addEdge(a, b);
        }
    }

    // DFS
    g.visited.assign(vertices, false);
    auto start = high_resolution_clock::now();
    cout << "\nDepth First Search:\n";
    g.dfs(0);  // Start DFS from vertex 0
    auto end = high_resolution_clock::now();
    double seq_dfs_time = duration<double>(end - start).count();
    cout << "\nTime taken: " << seq_dfs_time << " seconds\n";

    // Parallel DFS
    g.visited.assign(vertices, false);
    start = high_resolution_clock::now();
    cout << "\nParallel Depth First Search:\n";
    g.parallel_dfs(0);  // Start parallel DFS from vertex 0
    end = high_resolution_clock::now();
    double par_dfs_time = duration<double>(end - start).count();
    cout << "\nTime taken: " << par_dfs_time << " seconds\n";
    cout << "Speedup (DFS): " << seq_dfs_time / par_dfs_time << "\n";

    // BFS
    g.visited.assign(vertices, false);
    start = high_resolution_clock::now();
    cout << "\nBreadth First Search:\n";
    g.bfs(0);  // Start BFS from vertex 0
    end = high_resolution_clock::now();
    double seq_bfs_time = duration<double>(end - start).count();
    cout << "\nTime taken: " << seq_bfs_time << " seconds\n";

    // Parallel BFS
    g.visited.assign(vertices, false);
    start = high_resolution_clock::now();
    cout << "\nParallel Breadth First Search:\n";
    g.parallel_bfs(0);  // Start parallel BFS from vertex 0
    end = high_resolution_clock::now();
    double par_bfs_time = duration<double>(end - start).count();
    cout << "\nTime taken: " << par_bfs_time << " seconds\n";
    cout << "Speedup (BFS): " << seq_bfs_time / par_bfs_time << "\n";

    return 0;
}

