#ifndef HIDDENLAYER_
#define HIDDENLAYER_

#include "node.h"
#include "trsFuncMap.h"

template <typename T>
class hiddenLayer {
public:
	int numInputs;                          // number of input value, i.e. number of nodes in previous layer
	int numNodes;                           // number of nodes in this layer
	vector<node<T>> nodes;                  // array of nodes
	trs_func_type<batchData<T>> transFcn;       // transfer function between activation and output of all nodes
	trs_func_type<batchData<T>> transDrv;       // derivative of transfer function
	vector<batchData<T>> linearOutput;      // linear output of the layer before applying nonlinear activation function
public:
	// Constructor
	hiddenLayer(int _numInputs = 0, int _numNodes = 0, trs_func_type<batchData<T>> _transFcn = NULL, trs_func_type<batchData<T>> _transDrv = NULL) \
	: numInputs(_numInputs), numNodes(_numNodes), transFcn(_transFcn), transDrv(_transDrv)
    {
		for (int i = 0; i < numNodes; i++) 
			nodes.push_back(node<T>(numInputs));
		linearOutput = vector<batchData<T>>(numNodes, batchData<T>(0));
	}
    
    // Constructor with vector of trs_func_type
    hiddenLayer(int _numInputs, int _numNodes, vector<trs_func_type<batchData<T>>> transFcnInfo) \
    : numInputs(_numInputs), numNodes(_numNodes)
    {
        assert(transFcnInfo.size() == 2);
        transFcn = transFcnInfo[0];
        transDrv = transFcnInfo[1];
        nodes.resize(numNodes);
#pragma omp parallel for
        for (int i = 0; i < numNodes; i++)
            nodes[i] = node<T>(numInputs);
        linearOutput = vector<batchData<T>>(numNodes, batchData<T>(0));
    }

	// Destructor
	virtual ~hiddenLayer()
    {
		vector<node<T>>().swap(nodes);
		vector<batchData<T>>().swap(linearOutput);
	}
    
    // = overload
    hiddenLayer<T>& operator=(const hiddenLayer<T>& other) {
        if (this != &other) {
            numInputs = other.numInputs;
            numNodes = other.numNodes;
            nodes = other.nodes;
            transFcn = other.transFcn;
            transDrv = other.transDrv;
            linearOutput = other.linearOutput;
        }
        return *this;
    }

	// Assign weights for incoming edges of all nodes
	void setParams(vector<vector<T>> weights, vector<T> bias)
    {
#pragma omp parallel for
		for (int i = 0; i < numNodes; i++){
			nodes[i].setParams(weights[i],bias[i]);
		}
	}

	// Compute the output of all nodes
	vector<batchData<T>> computeOutputs(const vector<batchData<T>>& inputs)
    {
		vector<batchData<T>> ans(numNodes,batchData<T>(inputs[0].size()));
#pragma omp parallel for
		for (int i = 0; i < numNodes; i++){
			linearOutput[i] = nodes[i].computeLinearOutput(inputs);
			ans[i] = transFcn(linearOutput[i]);
		}
		return ans;
	}
	
	// Get output after execute computeOutputs
	vector<batchData<T>> getOutputs()
    {
		vector<batchData<T>> ans(linearOutput);
#pragma omp parallel for
		for (int i = 0; i < numNodes; i++){
			ans[i] = transFcn(linearOutput[i]);
		}
		return ans;
	}
	
	// Get output_drv after execute computeOutputs
	vector<batchData<T>> getOutputs_drv()
    {
		vector<batchData<T>> ans(linearOutput);
#pragma omp parallel for
		for (int i = 0; i < numNodes; i++){
			ans[i] = transDrv(linearOutput[i]);
		}
		return ans;
	}	

	// Update parameter
	void updateParameter(float epsilon)
	{
#pragma omp parallel for
		for (int i = 0; i < numNodes; i++)
			nodes[i].adaGradUpdateParameter(epsilon);
	}
};

#endif