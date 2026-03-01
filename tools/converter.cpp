#include "../shared/globals.hpp"
#include <cctype>


bool IsWeightedFormat(string format)
{
	if((format == "bwcsr")	||
		(format == "wcsr")	||
		(format == "wel"))
			return true;
	return false;
}

string GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

string ReplaceExtension(string fileName, string extension)
{
	size_t dot = fileName.find_last_of(".");
	if(dot == string::npos)
		return fileName + "." + extension;
	return fileName.substr(0, dot + 1) + extension;
}

bool IsCommentOrEmptyLine(const string& line)
{
	for(size_t i = 0; i < line.size(); i++)
	{
		unsigned char c = static_cast<unsigned char>(line[i]);
		if(!isspace(c))
			return c == '#' || c == '%';
	}
	return true;
}

int main(int argc, char** argv)
{
	if(argc!= 2)
	{
		cout << "\nThere was an error parsing command line arguments\n";
		exit(0);
	}
	
	string input = string(argv[1]);
	
	if(GetFileExtension(input) == "el" ||
		GetFileExtension(input) == "txt" ||
		GetFileExtension(input) == "snap")
	{
		ifstream infile;
		infile.open(input);
		stringstream ss;
		uint max = 0;
		string line;
		uint edgeCounter = 0;
		
		bool hasWeight = false;
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
			if(ss >> newEdge.w8)
				hasWeight = true;
			else
				newEdge.w8 = 1;
			
			edges.push_back(newEdge);
			edgeCounter++;
			
			if(max < newEdge.source)
				max = newEdge.source;
			if(max < newEdge.end)
				max = newEdge.end;				
		}			
		infile.close();
		
		uint num_nodes = max + 1;
		uint num_edges = edgeCounter;
		uint *nodePointer = new uint[num_nodes+1];
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
		uint *outDegreeCounter  = new uint[num_nodes];
		for(uint i=0; i<num_nodes; i++)
			outDegreeCounter[i] = 0;

		if(hasWeight)
		{
			OutEdgeWeighted *edgeList = new OutEdgeWeighted[num_edges];
			uint location;  
			for(uint i=0; i<num_edges; i++)
			{
				uint location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
				edgeList[location].end = edges[i].end;
				edgeList[location].w8 = edges[i].w8;
				outDegreeCounter[edges[i].source]++;  
			}

			std::ofstream outfile(ReplaceExtension(input, "bwcsr"), std::ofstream::binary);
			
			outfile.write((char*)&num_nodes, sizeof(unsigned int));
			outfile.write((char*)&num_edges, sizeof(unsigned int));
			outfile.write ((char*)nodePointer, sizeof(unsigned int)*num_nodes);
			outfile.write ((char*)edgeList, sizeof(OutEdgeWeighted)*num_edges);
			
			outfile.close();
			delete[] edgeList;
		}
		else
		{
			OutEdge *edgeList = new OutEdge[num_edges];
			uint location;  
			for(uint i=0; i<num_edges; i++)
			{
				uint location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
				edgeList[location].end = edges[i].end;
				outDegreeCounter[edges[i].source]++;  
			}

			std::ofstream outfile(ReplaceExtension(input, "bcsr"), std::ofstream::binary);
			
			outfile.write((char*)&num_nodes, sizeof(unsigned int));
			outfile.write((char*)&num_edges, sizeof(unsigned int));
			outfile.write ((char*)nodePointer, sizeof(unsigned int)*num_nodes);
			outfile.write ((char*)edgeList, sizeof(OutEdge)*num_edges);
			
			outfile.close();
			delete[] edgeList;
		}

		edges.clear();
		delete[] degree;
		delete[] outDegreeCounter;
	}
	else if(GetFileExtension(input) == "wel")
	{
		ifstream infile;
		infile.open(input);
		stringstream ss;
		uint max = 0;
		string line;
		uint edgeCounter = 0;
		
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
		
		uint num_nodes = max + 1;
		uint num_edges = edgeCounter;
		uint *nodePointer = new uint[num_nodes+1];
		OutEdgeWeighted *edgeList = new OutEdgeWeighted[num_edges];
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
		uint *outDegreeCounter  = new uint[num_nodes];
		for(uint i=0; i<num_nodes; i++)
			outDegreeCounter[i] = 0;
		uint location;  
		for(uint i=0; i<num_edges; i++)
		{
			uint location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
			edgeList[location].end = edges[i].end;
			edgeList[location].w8 = edges[i].w8;
			outDegreeCounter[edges[i].source]++;  
		}
		edges.clear();
		delete[] degree;
		delete[] outDegreeCounter;
		
		std::ofstream outfile(ReplaceExtension(input, "bwcsr"), std::ofstream::binary);
		
		outfile.write((char*)&num_nodes, sizeof(unsigned int));
		outfile.write((char*)&num_edges, sizeof(unsigned int));
		outfile.write ((char*)nodePointer, sizeof(unsigned int)*num_nodes);
		outfile.write ((char*)edgeList, sizeof(OutEdgeWeighted)*num_edges);
		
		outfile.close();
	}
	else
	{
		cout << "\nInput file format is not supported.\n";
		exit(0);
	}

}
