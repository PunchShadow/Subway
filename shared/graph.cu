#include "graph.cuh"
#include "gpu_error_check.cuh"
#include <cctype>

static bool IsCommentOrEmptyLine(const string& line)
{
	for(size_t i = 0; i < line.size(); i++)
	{
		unsigned char c = static_cast<unsigned char>(line[i]);
		if(!isspace(c))
			return c == '#' || c == '%';
	}
	return true;
}

template <class E>
Graph<E>::Graph(string graphFilePath, bool isWeighted)
{
	this->graphFilePath = graphFilePath;
	this->isWeighted = isWeighted;
}

template <class E>
string Graph<E>::GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

template <>
void Graph<OutEdgeWeighted>::AssignW8(uint w8, uint index)
{
    edgeList[index].w8 = w8;
}

template <>
void Graph<OutEdge>::AssignW8(uint w8, uint index)
{
    edgeList[index].end = edgeList[index].end; // do nothing
}

template <class E>
void Graph<E>::ReadGraph()
{

	cout << "Reading the input graph from the following file:\n>> " << graphFilePath << endl;
	
	this->graphFormat = GetFileExtension(graphFilePath);
	
	if(graphFormat == "bcsr" || graphFormat == "bwcsr")
	{
		ifstream infile (graphFilePath, ios::in | ios::binary);
		bool fileHasWeight = (graphFormat == "bwcsr");
	
		infile.read ((char*)&num_nodes, sizeof(uint));
		infile.read ((char*)&num_edges, sizeof(uint));
		
		nodePointer = new uint[num_nodes+1];
		gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
		
		infile.read ((char*)nodePointer, sizeof(uint)*num_nodes);
		nodePointer[num_nodes] = num_edges;

		if(fileHasWeight)
		{
			vector<OutEdgeWeighted> weightedEdgeList(num_edges);
			if(num_edges > 0)
				infile.read((char*)weightedEdgeList.data(), sizeof(OutEdgeWeighted) * num_edges);
			for(uint i = 0; i < num_edges; i++)
			{
				edgeList[i].end = weightedEdgeList[i].end;
				AssignW8(weightedEdgeList[i].w8, i);
			}
		}
		else
		{
			vector<OutEdge> plainEdgeList(num_edges);
			if(num_edges > 0)
				infile.read((char*)plainEdgeList.data(), sizeof(OutEdge) * num_edges);
			for(uint i = 0; i < num_edges; i++)
			{
				edgeList[i].end = plainEdgeList[i].end;
				AssignW8(1, i); // default weight for weighted apps reading .bcsr
			}
		}
	}
	else if(graphFormat == "el" || graphFormat == "wel" ||
		graphFormat == "txt" || graphFormat == "snap")
	{
		ifstream infile;
		infile.open(graphFilePath);
		stringstream ss;
		uint max = 0;
		string line;
		uint edgeCounter = 0;
		if(isWeighted)
		{
			vector<EdgeWeighted> edges;
			EdgeWeighted newEdge;
			while(getline( infile, line ))
			{
				if(IsCommentOrEmptyLine(line))
					continue;

				ss.str("");
				ss.clear();
				ss << line;

				if(!(ss >> newEdge.source >> newEdge.end))
					continue;
				if(!(ss >> newEdge.w8))
					newEdge.w8 = 1;
				
				edges.push_back(newEdge);
				edgeCounter++;
				
				if(max < newEdge.source)
					max = newEdge.source;
				if(max < newEdge.end)
					max = newEdge.end;				
			}
			infile.close();
			num_nodes = max + 1;
			num_edges = edgeCounter;
			nodePointer = new uint[num_nodes+1];
			gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
			uint *degree = new uint[num_nodes];
			for(uint i=0; i<num_nodes; i++)
				degree[i] = 0;
			for(uint i=0; i<num_edges; i++)
				degree[edges[i].source]++;
			
			uint counter=0;
			for(uint i=0; i<num_nodes; i++)
			{
				nodePointer[i] = counter;
				counter = counter + degree[i];
			}
			nodePointer[num_nodes] = num_edges;
			uint *outDegreeCounter  = new uint[num_nodes];
			for(uint i=0; i<num_nodes; i++)
				outDegreeCounter[i] = 0;
			uint location;  
			for(uint i=0; i<num_edges; i++)
			{
				location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
				edgeList[location].end = edges[i].end;
				if(isWeighted)
					AssignW8(edges[i].w8, location);
					//edgeList[location].w8 = edges[i].w8;
				outDegreeCounter[edges[i].source]++;  
			}
			edges.clear();
			delete[] degree;
			delete[] outDegreeCounter;
			
		}
		else
		{
			vector<Edge> edges;
			Edge newEdge;
			while(getline( infile, line ))
			{
				if(IsCommentOrEmptyLine(line))
					continue;

				ss.str("");
				ss.clear();
				ss << line;

				if(!(ss >> newEdge.source >> newEdge.end))
					continue;
				
				edges.push_back(newEdge);
				edgeCounter++;
				
				if(max < newEdge.source)
					max = newEdge.source;
				if(max < newEdge.end)
					max = newEdge.end;				
			}
			infile.close();
			num_nodes = max + 1;
			num_edges = edgeCounter;
			nodePointer = new uint[num_nodes+1];
			gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
			uint *degree = new uint[num_nodes];
			for(uint i=0; i<num_nodes; i++)
				degree[i] = 0;
			for(uint i=0; i<num_edges; i++)
				degree[edges[i].source]++;
			
			uint counter=0;
			for(uint i=0; i<num_nodes; i++)
			{
				nodePointer[i] = counter;
				counter = counter + degree[i];
			}
			nodePointer[num_nodes] = num_edges;
			uint *outDegreeCounter  = new uint[num_nodes];
			for(uint i=0; i<num_nodes; i++)
				outDegreeCounter[i] = 0;
			uint location;  
			for(uint i=0; i<num_edges; i++)
			{
				location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
				edgeList[location].end = edges[i].end;
				//if(isWeighted)
				//	edgeList[location].w8 = edges[i].w8;
				outDegreeCounter[edges[i].source]++;  
			}
			edges.clear();
			delete[] degree;
			delete[] outDegreeCounter;						
		}
	}
	else
	{
		cout << "The graph format is not supported!\n";
		exit(-1);
	}
	
	outDegree  = new unsigned int[num_nodes];
	
	for(uint i=1; i<num_nodes-1; i++)
		outDegree[i-1] = nodePointer[i] - nodePointer[i-1];
	outDegree[num_nodes-1] = num_edges - nodePointer[num_nodes-1];
	
	label1 = new bool[num_nodes];
	label2 = new bool[num_nodes];
	value  = new unsigned int[num_nodes];
	
	gpuErrorcheck(cudaMalloc(&d_outDegree, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_value, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_label1, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label2, num_nodes * sizeof(bool)));
	
	cout << "Done reading.\n";
	cout << "Number of nodes = " << num_nodes << endl;
	cout << "Number of edges = " << num_edges << endl;


}

//--------------------------------------

template <class E>
GraphPR<E>::GraphPR(string graphFilePath, bool isWeighted)
{
	this->graphFilePath = graphFilePath;
	this->isWeighted = isWeighted;
}

template <class E>
string GraphPR<E>::GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

template <>
void GraphPR<OutEdgeWeighted>::AssignW8(uint w8, uint index)
{
    edgeList[index].w8 = w8;
}

template <>
void GraphPR<OutEdge>::AssignW8(uint w8, uint index)
{
    edgeList[index].end = edgeList[index].end; // do nothing
}

template <class E>
void GraphPR<E>::ReadGraph()
{

	cout << "Reading the input graph from the following file:\n>> " << graphFilePath << endl;
	
	this->graphFormat = GetFileExtension(graphFilePath);
	
	if(graphFormat == "bcsr" || graphFormat == "bwcsr")
	{
		ifstream infile (graphFilePath, ios::in | ios::binary);
		bool fileHasWeight = (graphFormat == "bwcsr");
	
		infile.read ((char*)&num_nodes, sizeof(uint));
		infile.read ((char*)&num_edges, sizeof(uint));
		
		nodePointer = new uint[num_nodes+1];
		gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
		
		infile.read ((char*)nodePointer, sizeof(uint)*num_nodes);
		nodePointer[num_nodes] = num_edges;

		if(fileHasWeight)
		{
			vector<OutEdgeWeighted> weightedEdgeList(num_edges);
			if(num_edges > 0)
				infile.read((char*)weightedEdgeList.data(), sizeof(OutEdgeWeighted) * num_edges);
			for(uint i = 0; i < num_edges; i++)
			{
				edgeList[i].end = weightedEdgeList[i].end;
				AssignW8(weightedEdgeList[i].w8, i);
			}
		}
		else
		{
			vector<OutEdge> plainEdgeList(num_edges);
			if(num_edges > 0)
				infile.read((char*)plainEdgeList.data(), sizeof(OutEdge) * num_edges);
			for(uint i = 0; i < num_edges; i++)
			{
				edgeList[i].end = plainEdgeList[i].end;
				AssignW8(1, i); // default weight for weighted apps reading .bcsr
			}
		}
	}
	else if(graphFormat == "el" || graphFormat == "wel" ||
		graphFormat == "txt" || graphFormat == "snap")
	{
		ifstream infile;
		infile.open(graphFilePath);
		stringstream ss;
		uint max = 0;
		string line;
		uint edgeCounter = 0;
		if(isWeighted)
		{
			vector<EdgeWeighted> edges;
			EdgeWeighted newEdge;
			while(getline( infile, line ))
			{
				if(IsCommentOrEmptyLine(line))
					continue;

				ss.str("");
				ss.clear();
				ss << line;

				if(!(ss >> newEdge.source >> newEdge.end))
					continue;
				if(!(ss >> newEdge.w8))
					newEdge.w8 = 1;
				
				edges.push_back(newEdge);
				edgeCounter++;
				
				if(max < newEdge.source)
					max = newEdge.source;
				if(max < newEdge.end)
					max = newEdge.end;				
			}
			infile.close();
			num_nodes = max + 1;
			num_edges = edgeCounter;
			nodePointer = new uint[num_nodes+1];
			gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
			uint *degree = new uint[num_nodes];
			for(uint i=0; i<num_nodes; i++)
				degree[i] = 0;
			for(uint i=0; i<num_edges; i++)
				degree[edges[i].source]++;
			
			uint counter=0;
			for(uint i=0; i<num_nodes; i++)
			{
				nodePointer[i] = counter;
				counter = counter + degree[i];
			}
			nodePointer[num_nodes] = num_edges;
			uint *outDegreeCounter  = new uint[num_nodes];
			for(uint i=0; i<num_nodes; i++)
				outDegreeCounter[i] = 0;
			uint location;  
			for(uint i=0; i<num_edges; i++)
			{
				location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
				edgeList[location].end = edges[i].end;
				if(isWeighted)
					AssignW8(edges[i].w8, location);
					//edgeList[location].w8 = edges[i].w8;
				outDegreeCounter[edges[i].source]++;  
			}
			edges.clear();
			delete[] degree;
			delete[] outDegreeCounter;
			
		}
		else
		{
			vector<Edge> edges;
			Edge newEdge;
			while(getline( infile, line ))
			{
				if(IsCommentOrEmptyLine(line))
					continue;

				ss.str("");
				ss.clear();
				ss << line;

				if(!(ss >> newEdge.source >> newEdge.end))
					continue;
				
				edges.push_back(newEdge);
				edgeCounter++;
				
				if(max < newEdge.source)
					max = newEdge.source;
				if(max < newEdge.end)
					max = newEdge.end;				
			}
			infile.close();
			num_nodes = max + 1;
			num_edges = edgeCounter;
			nodePointer = new uint[num_nodes+1];
			gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
			uint *degree = new uint[num_nodes];
			for(uint i=0; i<num_nodes; i++)
				degree[i] = 0;
			for(uint i=0; i<num_edges; i++)
				degree[edges[i].source]++;
			
			uint counter=0;
			for(uint i=0; i<num_nodes; i++)
			{
				nodePointer[i] = counter;
				counter = counter + degree[i];
			}
			nodePointer[num_nodes] = num_edges;
			uint *outDegreeCounter  = new uint[num_nodes];
			for(uint i=0; i<num_nodes; i++)
				outDegreeCounter[i] = 0;
			uint location;  
			for(uint i=0; i<num_edges; i++)
			{
				location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
				edgeList[location].end = edges[i].end;
				//if(isWeighted)
				//	edgeList[location].w8 = edges[i].w8;
				outDegreeCounter[edges[i].source]++;  
			}
			edges.clear();
			delete[] degree;
			delete[] outDegreeCounter;						
		}
	}
	else
	{
		cout << "The graph format is not supported!\n";
		exit(-1);
	}
	
	outDegree  = new unsigned int[num_nodes];
	
	for(uint i=1; i<num_nodes-1; i++)
		outDegree[i-1] = nodePointer[i] - nodePointer[i-1];
	outDegree[num_nodes-1] = num_edges - nodePointer[num_nodes-1];
	

	value  = new float[num_nodes];
	delta  = new float[num_nodes];
	
	gpuErrorcheck(cudaMalloc(&d_outDegree, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_value, num_nodes * sizeof(float)));
	gpuErrorcheck(cudaMalloc(&d_delta, num_nodes * sizeof(float)));
	
	
	cout << "Done reading.\n";
	cout << "Number of nodes = " << num_nodes << endl;
	cout << "Number of edges = " << num_edges << endl;
	

}


template class Graph<OutEdge>;
template class Graph<OutEdgeWeighted>;

template class GraphPR<OutEdge>;
template class GraphPR<OutEdgeWeighted>;
