#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>

using namespace std;

void parallelDFS(const vector<vector<int>> &graph, int start, vector<bool> &visited)
{
    stack<int> s;
    s.push(start);

#pragma omp parallel
    {
        vector<bool> local_visited(visited.size(), false);

#pragma omp single nowait
        {
            while (!s.empty())
            {
                int node;

#pragma omp critical
                {
                    if (!s.empty())
                    {
                        node = s.top();
                        s.pop();
                    }
                    else
                    {
                        node = -1;
                    }
                }

                if (node != -1 && !visited[node])
                {
#pragma omp critical
                    visited[node] = true;

                    cout << "Visited: " << node << " by thread " << omp_get_thread_num() << endl;

                    for (int neighbor : graph[node])
                    {
                        if (!visited[neighbor])
                        {
#pragma omp critical
                            s.push(neighbor);
                        }
                    }
                }
            }
        }
    }
}

int main()
{
    // Example: Undirected graph
    int V = 6; // number of vertices
    vector<vector<int>> graph(V);

    // Create the undirected graph
    graph[0] = {1, 2};
    graph[1] = {0, 3, 4};
    graph[2] = {0, 5};
    graph[3] = {1};
    graph[4] = {1, 5};
    graph[5] = {2, 4};

    vector<bool> visited(V, false);
    cout << "Starting Parallel DFS:\n";
    parallelDFS(graph, 0, visited);

    return 0;
}

// g++ -fopenmp parallel_dfs.cpp -o parallel_dfs
// ./parallel_dfs
