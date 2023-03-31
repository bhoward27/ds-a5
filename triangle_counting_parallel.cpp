// NOTE: This is just a copy of my solution for assignment 4 currently.
#include <mpi.h>
#include <cstdio>
#include "core/utils.h"
#include "core/graph.h"

long countTriangles(uintV *array1, uintE len1, uintV *array2, uintE len2, uintV u, uintV v)
{
    uintE i = 0, j = 0; // indexes for array1 and array2
    long count = 0;

    if (u == v)
        return count;

    while ((i < len1) && (j < len2)) {
        if (array1[i] == array2[j]) {
            if ((array1[i] != u) && (array1[i] != v)) {
                count++;
            } else {
                // triangle with self-referential edge -> ignore
            }
            i++;
            j++;
        } else if (array1[i] < array2[j]) {
            i++;
        } else {
            j++;
        }
    }
    return count;
}

void triangleCountParallelStrat1(Graph &g, int world_size, int world_rank)
{
    uintV n = g.n_;
    uintV m = g.m_;
    long global_count = 0;
    double time_taken;
    timer t1;
    t1.start();

    // Determine subsets of vertices to work on per process.
    uintV start_vertex = 0;
    uintV end_vertex = 0;
    for (int i = 0; i < world_size; i++) {
        start_vertex = end_vertex;
        long count = 0;
        while (end_vertex < n) {
            count += g.vertices_[end_vertex].getOutDegree();
            end_vertex++;
            if (count >= m / world_size) {
                break;
            }
        }
        if (i == world_rank) {
            break;
        }
    }

    // Count triangles.
    long local_count = 0;
    long edges_processed = 0;
    for (uintV u = start_vertex; u < end_vertex; u++) {
        uintE out_degree = g.vertices_[u].getOutDegree();
        edges_processed += out_degree;
        for (uintE i = 0; i < out_degree; i++) {
            uintV v = g.vertices_[u].getOutNeighbor(i);
            local_count += countTriangles(g.vertices_[u].getInNeighbors(),
                                          g.vertices_[u].getInDegree(),
                                          g.vertices_[v].getOutNeighbors(),
                                          g.vertices_[v].getOutDegree(),
                                          u,
                                          v);
        }
    }

    // Synchronization phase start.
    timer t2;
    t2.start();
    if (world_rank == 0) {
        // Add up the local counts from all processes.
        for (int i = 1; i < world_size; i++) {
            long count;
            MPI_Recv(&count, 1, MPI_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global_count += count;
        }
        global_count += local_count;
    }
    else {
        MPI_Send(&local_count, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
    }
    double communication_time = t2.stop();
    // Synchronization phase end.

    time_taken = t1.stop();
    if (world_rank == 0) {
        std::printf("rank, edges, triangle_count, communication_time\n");
    }

    // For every thread, print out the following statistics:
    // rank, edges, triangle_count, communication_time
    // 0, 17248443, 144441858, 0.000074
    // 1, 17248443, 152103585, 0.000020
    // 2, 17248443, 225182666, 0.000034
    // 3, 17248444, 185596640, 0.000022
    std::printf("%d, %ld, %ld, %f\n", world_rank, edges_processed, local_count, communication_time);



    if (world_rank == 0) {
        // Print out overall statistics
        std::printf("Number of triangles : %ld\n", global_count);
        std::printf("Number of unique triangles : %ld\n", global_count / 3);
        std::printf("Time taken (in seconds) : %f\n", time_taken);
    }
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("triangle_counting_serial", "Count the number of triangles using serial and parallel execution");
    options.add_options("custom", {
            {"strategy", "Strategy to be used", cxxopts::value<uint>()->default_value(DEFAULT_STRATEGY)},
            {"inputFile", "Input graph file path", cxxopts::value<std::string>()->default_value("/scratch/input_graphs/roadNet-CA")},
    });

    auto cl_options = options.parse(argc, argv);
    uint strategy = cl_options["strategy"].as<uint>();
    std::string input_file_path = cl_options["inputFile"].as<std::string>();

    MPI_Init(NULL, NULL);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank == 0) {
        std::printf("World size : %d\n", world_size);
        std::printf("Communication strategy : %d\n", strategy);
    }

    Graph g;
    g.readGraphFromBinary<int>(input_file_path);

    switch (strategy) {
        case 1:
            triangleCountParallelStrat1(g, world_size, world_rank);
            break;
        default:
            std::printf("Strategy %d is not implemented.\n", strategy);
            break;
    }

    MPI_Finalize();

    return 0;
}

