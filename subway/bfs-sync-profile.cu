// Instrumented BFS-sync that outputs per-iteration timing CSV:
//   gItr, numActiveNodes, numActiveEdges, cpuPackingMs, h2dMs, gpuComputeMs

#include "../shared/globals.hpp"
#include "../shared/timer.hpp"
#include "../shared/argument_parsing.cuh"
#include "../shared/graph.cuh"
#include "../shared/subgraph.cuh"
#include "../shared/partitioner.cuh"
#include "../shared/subgraph_generator.cuh"
#include "../shared/gpu_error_check.cuh"
#include "../shared/gpu_kernels.cuh"
#include "../shared/subway_utilities.hpp"
#include <chrono>
#include <fstream>

using hrc = std::chrono::high_resolution_clock;

static double elapsed_ms(hrc::time_point a, hrc::time_point b) {
	return std::chrono::duration<double, std::milli>(b - a).count();
}

template <class E>
static double generate_and_measure_packing(SubgraphGenerator<E> &subgen,
                                            Graph<E> &graph,
                                            Subgraph<E> &subgraph)
{
	auto t0 = hrc::now();
	subgen.generate(graph, subgraph);
	cudaDeviceSynchronize();
	auto t1 = hrc::now();
	return elapsed_ms(t0, t1);
}

int main(int argc, char** argv)
{
	cudaFree(0);

	ArgumentParser arguments(argc, argv, true, false);

	Timer timer;
	timer.Start();

	Graph<OutEdge> graph(arguments.input, false);
	graph.ReadGraph();

	float readtime = timer.Finish();
	cout << "Graph Reading finished in " << readtime/1000 << " (s).\n";

	for(unsigned int i=0; i<graph.num_nodes; i++)
	{
		graph.value[i] = DIST_INFINITY;
		graph.label1[i] = false;
		graph.label2[i] = false;
	}
	graph.value[arguments.sourceNode] = 0;
	graph.label1[arguments.sourceNode] = false;
	graph.label2[arguments.sourceNode] = true;

	gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_value, graph.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_label1, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_label2, graph.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));

	SubgraphGenerator<OutEdge> subgen(graph);
	Subgraph<OutEdge> subgraph(graph.num_nodes, graph.num_edges);

	subgen.generate(graph, subgraph);

	Partitioner<OutEdge> partitioner;

	// CSV output
	string csvPath = "bfs_sync_profile_output.csv";
	ofstream csv(csvPath);
	csv << "gItr,numActiveNodes,numActiveEdges,cpuPackingMs,h2dMs,gpuComputeMs\n";

	timer.Start();

	uint itr = 0;

	while (subgraph.numActiveNodes > 0)
	{
		itr++;

		partitioner.partition(subgraph, subgraph.numActiveNodes);

		double h2d_total_ms = 0.0;
		double gpu_compute_total_ms = 0.0;

		for(int i=0; i<partitioner.numPartitions; i++)
		{
			// --- H2D Transfer ---
			cudaDeviceSynchronize();
			auto h2d_start = hrc::now();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList,
			                         subgraph.activeEdgeList + partitioner.fromEdge[i],
			                         (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge),
			                         cudaMemcpyHostToDevice));
			cudaDeviceSynchronize();
			auto h2d_end = hrc::now();
			h2d_total_ms += elapsed_ms(h2d_start, h2d_end);

			// --- GPU Compute ---
			auto gpu_start = hrc::now();

			moveUpLabels<<<partitioner.partitionNodeSize[i]/512 + 1, 512>>>(
				subgraph.d_activeNodes, graph.d_label1, graph.d_label2,
				partitioner.partitionNodeSize[i], partitioner.fromNode[i]);

			bfs_kernel<<<partitioner.partitionNodeSize[i]/512 + 1, 512>>>(
				partitioner.partitionNodeSize[i],
				partitioner.fromNode[i],
				partitioner.fromEdge[i],
				subgraph.d_activeNodes,
				subgraph.d_activeNodesPointer,
				subgraph.d_activeEdgeList,
				graph.d_outDegree,
				graph.d_value,
				graph.d_label1,
				graph.d_label2);

			cudaDeviceSynchronize();
			gpuErrorcheck(cudaPeekAtLastError());

			auto gpu_end = hrc::now();
			gpu_compute_total_ms += elapsed_ms(gpu_start, gpu_end);
		}

		// Compute numActiveEdges for this iteration
		unsigned int numActiveEdges = 0;
		if(subgraph.numActiveNodes > 0)
			numActiveEdges = subgraph.activeNodesPointer[subgraph.numActiveNodes-1]
			                 + graph.outDegree[subgraph.activeNodes[subgraph.numActiveNodes-1]];

		unsigned int activeNodesBefore = subgraph.numActiveNodes;

		// --- CPU Packing (subgraph generation for NEXT iteration) ---
		double cpu_packing_ms = generate_and_measure_packing(subgen, graph, subgraph);

		csv << itr << ","
		    << activeNodesBefore << ","
		    << numActiveEdges << ","
		    << cpu_packing_ms << ","
		    << h2d_total_ms << ","
		    << gpu_compute_total_ms << "\n";

		cout << "[Profile] Iter " << itr
		     << " | ActiveNodes=" << activeNodesBefore
		     << " | ActiveEdges=" << numActiveEdges
		     << " | CpuPack=" << cpu_packing_ms << "ms"
		     << " | H2D=" << h2d_total_ms << "ms"
		     << " | GPU=" << gpu_compute_total_ms << "ms" << endl;
	}

	csv.close();
	cout << "Per-iteration CSV written to: " << csvPath << endl;

	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime/1000 << " (s).\n";
	cout << "Number of iterations = " << itr << endl;

	gpuErrorcheck(cudaMemcpy(graph.value, graph.d_value, graph.num_nodes*sizeof(uint), cudaMemcpyDeviceToHost));
	utilities::PrintResults(graph.value, min(30, graph.num_nodes));

	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, graph.value, graph.num_nodes);
}
