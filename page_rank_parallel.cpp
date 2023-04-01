// NOTE: This is just a copy of my solution for assignment 4 currently.
#include <mpi.h>
#include <cstdio>
#include <vector>

#include "core/utils.h"
#include "core/graph.h"

#ifdef USE_INT
#define INIT_PAGE_RANK 100000
#define EPSILON 1000
#define PAGE_RANK(x) (15000 + (5 * x) / 6)
#define CHANGE_IN_PAGE_RANK(x, y) std::abs(x - y)
#define PAGERANK_MPI_TYPE MPI_LONG
#define PR_FMT "%ld"
typedef int64_t PageRankType;
#define MPI_PR_TYPE MPI_LONG
#else
#define INIT_PAGE_RANK 1.0
#define EPSILON 0.01
#define DAMPING 0.85
#define PAGE_RANK(x) (1 - DAMPING + DAMPING * x)
#define CHANGE_IN_PAGE_RANK(x, y) std::fabs(x - y)
#define PAGERANK_MPI_TYPE MPI_FLOAT
#define PR_FMT "%f"
typedef float PageRankType;
#define MPI_PR_TYPE MPI_FLOAT
#endif

void pageRankParallelStrat1(Graph &g, int max_iters, int world_rank, int world_size)
{
    uintV n = g.n_;
    uintV m = g.m_;
    double time_taken;
    timer t1;
    PageRankType *pr_curr = new PageRankType[n];
    PageRankType *pr_next = new PageRankType[n];

    t1.start();
    for (uintV i = 0; i < n; i++) {
        pr_curr[i] = INIT_PAGE_RANK;
        pr_next[i] = 0.0;
    }

    // Determine subsets of vertices to work on per process.
    uintV start_vertex = 0;
    uintV end_vertex = 0;
    std::vector<uintV> vertex_boundaries; // Need for later to distribute vertex subsets from root to non-root processes.

    for (int i = 0; i < world_size; i++) {
        start_vertex = end_vertex;
        long count = 0;
        while (end_vertex < n) {
            count += g.vertices_[end_vertex].getOutDegree();
            end_vertex++;
            if (count >= m / world_size) {
                if (i <= world_size - 1) vertex_boundaries.push_back(end_vertex);
                break;
            }
        }
    }
    if (world_rank == 0) {
        start_vertex = 0;
        end_vertex = vertex_boundaries[0];
    }
    else if (world_rank < world_size - 1) {
        start_vertex = vertex_boundaries[world_rank - 1];
        end_vertex = vertex_boundaries[world_rank];
    }
    else {
        start_vertex = vertex_boundaries[world_rank - 1];
        end_vertex = n;
    }

    double communication_time = 0.0;
    long edges_processed = 0;
    for (int i = 0; i < max_iters; i++) {
        for (uintV u = start_vertex; u < end_vertex; u++) {
            uintE out_degree = g.vertices_[u].getOutDegree();
            edges_processed += out_degree;
            for (uintE i = 0; i < out_degree; i++) {
                uintV v = g.vertices_[u].getOutNeighbor(i);
                pr_next[v] += pr_curr[u] / out_degree;
            }
        }

        // Synchronization phase 1 start
        timer t2;
        t2.start();

        // The root process sums up the next_page_rank value for every vertex in the graph using MPI_Reduce().
        if (world_rank == 0) {
            MPI_Reduce(MPI_IN_PLACE, pr_next, n, MPI_PR_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Reduce(pr_next, pr_next, n, MPI_PR_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        // Then, the root process sends the aggregated next_page_rank value of each vertex to its appropriate process.
        //     For example, if vertex v is allocated to process P1, the aggregated next_page_rank[v] is sent only to P1.
        // The root process sends back the corresponding SUBSET of pr_next to the non-root processes.
        int* counts = nullptr;
        int* displacements = nullptr;
        if (world_rank == 0) {
            counts = new int[world_size];
            displacements = new int[world_size];
            uintV start_vertex = 0;
            uintV end_vertex = vertex_boundaries[0];
            uintV count = end_vertex - start_vertex;
            displacements[0] = start_vertex;
            counts[0] = count;
            // std::printf("count[%d] = %d\n", 0, count);
            // std::printf("displacements[%d] = %d\n", 0, start_vertex);
            for (int i = 1; i < world_size; i++) {
                uintV start = vertex_boundaries[i - 1];
                uintV end = (i < world_size - 1) ? vertex_boundaries[i] : n;
                uintV count = end - start;
                displacements[i] = start;
                counts[i] = count;
                // std::printf("count[%d] = %d\n", i, count);
                // std::printf("displacements[%d] = %d\n", i, start);
            }
        }

        uintV start, end, count;
        if (world_rank) {
            start = vertex_boundaries[world_rank - 1];
            end = (world_rank < world_size - 1) ? vertex_boundaries[world_rank] : n;
            count = end - start;
        }
        else {
            start = 0;
            end = vertex_boundaries[0];
            count = end - start;
        }
        if (world_rank == 0) {
            PageRankType* a = new PageRankType[n];
            MPI_Scatterv(pr_next,
                        counts,
                        displacements,
                        MPI_PR_TYPE,
                        a,
                        n,
                        MPI_PR_TYPE,
                        0,
                        MPI_COMM_WORLD);
            delete[] a;
            a = nullptr;
        }
        else {
            // std::printf("rank = %d\nn=%d\ncount = %d\n", world_rank, n, count);
            MPI_Scatterv(nullptr,
                        nullptr,
                        nullptr,
                        MPI_PR_TYPE,
                        &pr_next[start],
                        count,
                        MPI_PR_TYPE,
                        0,
                        MPI_COMM_WORLD);
        }


        if (world_rank == 0) {
            delete[] displacements;
            delete[] counts;
            displacements = nullptr;
            counts = nullptr;
        }

        communication_time += t2.stop();
        // Synchronization phase 1 end.

        for (uintV v = start_vertex; v < end_vertex; v++) {
            pr_curr[v] = PAGE_RANK(pr_next[v]);
        }
        for (uintV v = 0; v < n; v++) {
            pr_next[v] = 0;
        }
    }




    if (world_rank == 0) {
        std::printf("rank, num_edges, communication_time\n");
    }
    // For every thread, print the following statistics:
    // rank, num_edges, communication_time
    // 0, 344968860, 1.297778
    // 1, 344968860, 1.247763
    // 2, 344968860, 0.956243
    // 3, 344968880, 0.467028
    std::printf("%d, %ld, %f\n", world_rank, edges_processed, communication_time);

    PageRankType local_sum = 0;
    for (uintV v = start_vertex; v < end_vertex; v++) {
        local_sum += pr_curr[v];
    }

    PageRankType sum_of_page_ranks = 0;

    MPI_Reduce(&local_sum, &sum_of_page_ranks, 1, MPI_PR_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
    time_taken = t1.stop();
    if (world_rank == 0) {
        std::printf("Sum of page rank : " PR_FMT "\n", sum_of_page_ranks);
        std::printf("Time taken (in seconds) : %f\n", time_taken);
    }


    delete[] pr_curr;
    delete[] pr_next;
    pr_curr = nullptr;
    pr_next = nullptr;
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("page_rank_push", "Calculate page_rank using serial and parallel execution");
    options.add_options("", {
                                {"nIterations", "Maximum number of iterations", cxxopts::value<uint>()->default_value(DEFAULT_MAX_ITER)},
                                {"strategy", "Strategy to be used", cxxopts::value<uint>()->default_value(DEFAULT_STRATEGY)},
                                {"inputFile", "Input graph file path", cxxopts::value<std::string>()->default_value("/scratch/input_graphs/roadNet-CA")},
                            });

    auto cl_options = options.parse(argc, argv);
    uint strategy = cl_options["strategy"].as<uint>();
    uint max_iterations = cl_options["nIterations"].as<uint>();
    std::string input_file_path = cl_options["inputFile"].as<std::string>();

    MPI_Init(NULL, NULL);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank == 0) {
#ifdef USE_INT
        std::printf("Using INT\n");
#else
        std::printf("Using FLOAT\n");
#endif
        std::printf("World size : %d\n", world_size);
        std::printf("Communication strategy : %d\n", strategy);
        std::printf("Iterations : %d\n", max_iterations);
    }

    Graph g;
    g.readGraphFromBinary<int>(input_file_path);

    switch (strategy) {
        case 1:
            pageRankParallelStrat1(g, max_iterations, world_rank, world_size);
            break;
        default:
            std::printf("Strategy %d has not been implemented.\n", strategy);
            break;
    }

    MPI_Finalize();

    return 0;
}
