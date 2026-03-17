
#include "subgraph.cuh"
#include "gpu_error_check.cuh"
#include "graph.cuh"
#include <cuda_profiler_api.h>

namespace
{
const double kEdgeBufferFraction = 0.80;
const ull kMinDeviceHeadroomBytes = 512ULL * 1024ULL * 1024ULL;
}

template <class E>
Subgraph<E>::Subgraph(uint num_nodes, uint num_edges)
{
	this->num_nodes = num_nodes;
	this->num_edges = num_edges;
	
	gpuErrorcheck(cudaMallocHost(&activeNodes, num_nodes * sizeof(uint)));
	gpuErrorcheck(cudaMallocHost(&activeNodesPointer, (num_nodes+1) * sizeof(uint)));
	gpuErrorcheck(cudaMallocHost(&activeEdgeList, num_edges * sizeof(E)));
	
	gpuErrorcheck(cudaMalloc(&d_activeNodes, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_activeNodesPointer, (num_nodes+1) * sizeof(unsigned int)));

	size_t freeMemBytes = 0;
	size_t totalMemBytes = 0;
	gpuErrorcheck(cudaMemGetInfo(&freeMemBytes, &totalMemBytes));
	(void)totalMemBytes;

	// Leave room for later Thrust scans and CUDA runtime allocations.
	ull headroomBytes = static_cast<ull>(freeMemBytes * (1.0 - kEdgeBufferFraction));
	if(headroomBytes < kMinDeviceHeadroomBytes)
		headroomBytes = kMinDeviceHeadroomBytes;

	if(freeMemBytes <= headroomBytes)
	{
		cerr << "Insufficient free GPU memory after reserving headroom for CUDA/Thrust temporary storage." << endl;
		exit(-1);
	}

	ull availableEdgeBufferBytes = static_cast<ull>(freeMemBytes) - headroomBytes;
	max_partition_size = availableEdgeBufferBytes / sizeof(E);

	if(max_partition_size == 0)
	{
		cerr << "Insufficient free GPU memory for the active edge buffer." << endl;
		exit(-1);
	}

	if(max_partition_size > num_edges)
		max_partition_size = num_edges;
	if(max_partition_size > DIST_INFINITY)
		max_partition_size = DIST_INFINITY;

	gpuErrorcheck(cudaMalloc(&d_activeEdgeList, max_partition_size * sizeof(E)));
}

template class Subgraph<OutEdge>;
template class Subgraph<OutEdgeWeighted>;

// For initialization with one active node
//unsigned int numActiveNodes = 1;
//subgraph.activeNodes[0] = SOURCE_NODE;
//for(unsigned int i=graph.nodePointer[SOURCE_NODE], j=0; i<graph.nodePointer[SOURCE_NODE] + graph.outDegree[SOURCE_NODE]; i++, j++)
//	subgraph.activeEdgeList[j] = graph.edgeList[i];
//subgraph.activeNodesPointer[0] = 0;
//subgraph.activeNodesPointer[1] = graph.outDegree[SOURCE_NODE];
//gpuErrorcheck(cudaMemcpy(subgraph.d_activeNodes, subgraph.activeNodes, numActiveNodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
//gpuErrorcheck(cudaMemcpy(subgraph.d_activeNodesPointer, subgraph.activeNodesPointer, (numActiveNodes+1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
