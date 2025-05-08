#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

void parallelBFS(const vector<vector<int>>& graph, int start) {
    int n = graph.size();
    vector<bool> visited(n, false);
    vector<int> level(n, -1);

    queue<int> q;
    visited[start] = true;
    level[start] = 0;
    q.push(start);

    cout << "BFS traversal from node " << start << ":\n";

    while (!q.empty()) {
        int qSize = q.size();
        vector<int> currentLevel;

        // Copy current level nodes
        for (int i = 0; i < qSize; ++i) {
            int node = q.front();
            q.pop();
            currentLevel.push_back(node);
        }

        // Parallel processing of current level
        #pragma omp parallel for
        for (int i = 0; i < currentLevel.size(); ++i) {
            int u = currentLevel[i];
            #pragma omp critical
            cout << u << " ";

            for (int v : graph[u]) {
                if (!visited[v]) {
                    #pragma omp critical
                    {
                        if (!visited[v]) {
                            visited[v] = true;
                            level[v] = level[u] + 1;
                            q.push(v);
                        }
                    }
                }
            }
        }
    }
}

int main() {
    int V = 6;
    vector<vector<int>> graph(V);

    // Undirected graph
    graph[0] = {1, 2};
    graph[1] = {0, 3, 4};
    graph[2] = {0, 4};
    graph[3] = {1, 5};
    graph[4] = {1, 2, 5};
    graph[5] = {3, 4};

    double start_time = omp_get_wtime();
    parallelBFS(graph, 0);
    double end_time = omp_get_wtime();

    cout << "\nExecution time: " << end_time - start_time << " seconds\n";

    return 0;
}

// g++ -fopenmp parallel_bfs.cpp -o parallel_bfs
// ./parallel_bfs
