#ifndef NEURALNETWORK_
#define NEURALNETWORK_

#include <cstdlib>
#include <algorithm>
#include <ctime>
#include "hiddenLayer.h"
#include "objFuncMap.h"

// Struct of network batch input for neuralNetwork class
template<typename T>
struct networkBatchInput {
    vector<batchData<T>> input;
    vector<batchData<T>> target;
    networkBatchInput(size_t batchSize, size_t inputDim, size_t targetDim) {
        input = vector<batchData<T>>(inputDim, batchData<T>(batchSize));
        target = vector<batchData<T>>(targetDim, batchData<T>(batchSize));
    }
};

template <typename T>
class neuralNetwork {
private:
    int numInputs;                              // number of input dimension
	int numLayers;                              // number of all hidden layers in the network
    vector<vector<batchData<T>>> layerOutputs;  // Use to store output of each hidden layer
    objFuncInfo<T> objFunc;                     // Objective function and its partial derivative w.r.t. output
	vector<hiddenLayer<T>> hiddenLayers;        // array of hidden layers
	float epsilon;                              // Unit step size for computing adaptive step size
public:
// Constructor using transfer function name
    neuralNetwork(int _numInputs, int _numLayers, vector<int> numNodesPerLayers, vector<string> trsFuncName, string objFuncName, float _epsilon = 0.01) : numInputs(_numInputs), numLayers(_numLayers), epsilon(_epsilon)
    {
        assert(numLayers > 1);
        trsFunctionMap<T> trsFuncMap;
        // i-th hidden layer has numNodesPerLayers[i-1] input value and numNodesPerLayers[i] outputs
        for (int i = 0; i < numLayers; i++)
            hiddenLayers.push_back(hiddenLayer<T>((i==0) ? numInputs : numNodesPerLayers[i - 1], numNodesPerLayers[i], trsFuncMap[trsFuncName[i]][0], trsFuncMap[trsFuncName[i]][1]));
        objFunctionMap<T> objFuncMap;
        objFunc = objFuncMap[objFuncName];
        // Malloc memory for layer input and output
        layerOutputs.resize(numLayers);
    }
    
    // Constructor from file
    neuralNetwork(string filename)
    {
        this->loadFile(filename);
        // Malloc memory for layer input and output
        layerOutputs.resize(numLayers);
    }

	// Destructor
	virtual ~neuralNetwork()
    {
		vector<hiddenLayer<T>>().swap(hiddenLayers);
	}

	// Assign weights for all edges in the network
	void setParams(vector<vector<vector<T>>> weights, vector<vector<T>> bias)
    {
		for (int i = 0; i < numLayers; i++)
			hiddenLayers[i].setWeights(weights[i], bias[i]);
	}
    
    // Train the neural network and store training/testing MSE for each epoch
    vector<vector<T>> train(int epoch, int batchSize, const vector<vector<T>>& trainData, const vector<vector<T>>& trainTarget, const vector<vector<T>>& testData, const vector<vector<T>>& testTarget, bool print_info = true)
    {
        // Guarantee that the sample number for input and target are same
        assert( trainData.size() == trainTarget.size() );
        assert( testData.size() == testTarget.size() );
        
        size_t numTrainSample = trainTarget.size();
        size_t numTrainBatch = numTrainSample / batchSize;
        size_t numTestSample = testTarget.size();
        size_t numTestBatch = numTestSample / batchSize;

        // Initialize vector to store loss for each epoch
        vector<T> trainLoss(epoch,0);
        vector<T> testLoss(epoch,0);
        
        // Divide testing data into batches
        if (print_info)
            cout << "Permute testing samples and divide them into batches" << endl;
        vector<networkBatchInput<T>> testBatches = divide(testData, testTarget, batchSize);
        
        vector<batchData<T>> outputs;
        T temp = 0;
        double timer, time_elapse;
        for (int t = 0; t < epoch; t++) {
            // Randomly permute training data for stochastic gradient descent algorithm
            if (print_info)
                cout << "Permute training samples and divide them into batches" << endl;
            vector<networkBatchInput<T>> trainBatches = divide(trainData, trainTarget, batchSize);
            // Train the network using training data and keep track of training error
            for (int i = 0; i < numTrainBatch; i++) {
                timer = omp_get_wtime();
                feedForward(trainBatches[i].input);
                backPropagation(trainBatches[i].input, trainBatches[i].target);
                outputs = layerOutputs[numLayers-1];
                temp = objFunc.func(trainBatches[i].target, outputs);
                trainLoss[t] += temp;
                
                time_elapse = omp_get_wtime( ) - timer;
                if (print_info)
                    cout << "Epoch " << t << ", batch " << i << ", train current batch loss = " << temp / batchSize << ", used " << time_elapse << " sec" << endl;
            }
            // Test the network using testing data and keep track of testing error
            for (int i = 0; i < numTestBatch; i++) {
                timer = omp_get_wtime();
                feedForward(testBatches[i].input);
                outputs = layerOutputs[numLayers-1];
                temp = objFunc.func(testBatches[i].target, outputs);
                testLoss[t] += temp;
                
                time_elapse = omp_get_wtime( ) - timer;
                if (print_info)
                    cout << "Epoch " << t << ", batch " << i << ", test current batch loss = " << temp / batchSize << ", used " << time_elapse << " sec" << endl;
            }
            if (print_info)
                cout << "Epoch " << t << ", train loss = " << trainLoss[t] / trainData.size() << ",\t test loss = " << testLoss[t] / testData.size() << endl;
        }
        return {trainLoss, testLoss};
    }
    
    // Train the neural network and store training MSE, do not use testing data during training procedure
    vector<T> train(int epoch, int batchSize, const vector<vector<T>>& trainData, const vector<vector<T>>& trainTarget, bool print_info = true)
    {
        // Guarantee that the sample number for input and target are same
        assert( trainData.size() == trainTarget.size() );
        
        size_t numTrainSample = trainTarget.size();
        size_t numTrainBatch = numTrainSample / batchSize;
        
        // Initialize vector to store loss for each epoch
        vector<T> trainLoss(epoch,0);
        
        vector<batchData<T>> outputs;
        T temp = 0;
        double timer, time_elapse;
        for (int t = 0; t < epoch; t++) {
            // Randomly permute training data for stochastic gradient descent algorithm
            if (print_info)
                cout << "Permute training samples and divide them into batches" << endl;
            vector<networkBatchInput<T>> trainBatches = divide(trainData, trainTarget, batchSize);
            timer = omp_get_wtime();
            // Train the network using training data and keep track of training error
            for (int i = 0; i < numTrainBatch; i++) {
                timer = omp_get_wtime();
                feedForward(trainBatches[i].input);
                backPropagation(trainBatches[i].input, trainBatches[i].target);
                outputs = layerOutputs[numLayers-1];
                temp = objFunc.func(trainBatches[i].target, outputs);
                trainLoss[t] += temp;
                
                time_elapse = omp_get_wtime( ) - timer;
                if (print_info)
                    cout << "Epoch " << t << ", batch " << i << ", train current batch loss = " << temp / batchSize << ", used " << time_elapse << " sec" << endl;
            }
            if (print_info)
                cout << "Epoch " << t << ", train loss = " << trainLoss[t] / trainData.size() << endl;
        }
        return trainLoss;
    }

    
    // Given input data, use the neural network to generate the output
    vector<vector<T>> apply(const vector<vector<T>>& inputData, int batchSize = 100, bool print_info = true)
    {
        assert( inputData.size() > 0 );
        
        size_t numSample = inputData.size();
        size_t numBatch = numSample / batchSize;
        
        // Divide testing data into batches without permute
        vector<vector<T>> dummy_inputTarget(numSample);
        vector<networkBatchInput<T>> batches = divide(inputData, dummy_inputTarget, batchSize, false);
        if (print_info)
            cout << "Divided input data into " << numBatch << " batches" << endl;
        
        // Pass the input through network to get output
        double timer, time_elapse;
        vector<vector<batchData<T>>> outputs(numBatch);
        for (int i = 0; i < numBatch; i++) {
            timer = omp_get_wtime();
            feedForward(batches[i].input);
            outputs[i] = layerOutputs[numLayers-1];
            
            time_elapse = omp_get_wtime( ) - timer;
            if (print_info)
                cout << "Batch " << i << " out of " << numBatch << " finished computation" << ", used " << time_elapse << " sec" << endl;
        }
        
        // Convert vector<vector<batchData<T>>> back to vector<vector<T>>
        size_t outputDim = layerOutputs[numLayers-1].size();
        vector<vector<T>> outputData(numSample, vector<T>(outputDim));
        for (int i = 0; i < numBatch; i++)
            for (int b = 0; b < batchSize; b++)
                for (int j = 0; j < outputDim; j++)
                    outputData[i*batchSize+b][j] = outputs[i][j][b];
        if (print_info)
            cout << "Finished conversion from batches to 2D-array" << endl;
        return outputData;
    }
    
    // Save the network settings to file
    int saveFile(string filename)
    {
        ofstream outfile;
        open_file(outfile,filename.c_str());
        if (!outfile) {
            cerr<< "cannot open" << endl;
            return 1;
        }
        
        // Save general parameters for a neural network
        outfile << numInputs << " " << numLayers << " " << epsilon << endl;
        
        // Save objective function name
        objFunctionMap<T> objFuncMap;
        outfile << objFuncMap(objFunc.func) << endl;
        
        // Save detailed settings for a neural network, i.e. information of hidden layers
        trsFunctionMap<T> trsFuncMap;
        for (int l = 0; l < numLayers; l++) {
            outfile << "{ " << endl;
            // Save general parameters for a hidden layer
            outfile << "\t" << hiddenLayers[l].numInputs << " " << hiddenLayers[l].numNodes << " " << trsFuncMap(hiddenLayers[l].transFcn) << endl;
            // Save detailed settings for a hidden layer, i.e. information of nodes
            for (int i = 0; i < hiddenLayers[l].numNodes; i++) {
                outfile << "\t[ " << endl;
                // Save bias and cum_bias of the node
                outfile << "\t\t" << hiddenLayers[l].nodes[i].bias << " " << hiddenLayers[l].nodes[i].cum_bias << endl << "\t\t";
                // Save weights of the node
                for (int k = 0; k < hiddenLayers[l].numInputs; k++)
                    outfile << hiddenLayers[l].nodes[i].weights[k] << " ";
                outfile << endl << "\t\t";
                // Save cum_weights of the node
                for (int k = 0; k < hiddenLayers[l].numInputs; k++)
                    outfile << hiddenLayers[l].nodes[i].cum_weights[k] << " ";
                outfile << endl << "\t];" << endl;
            }
            outfile << "};" << endl << endl;
        }
        
        outfile.close();
        return 0;
    }
    
    // Load the network settings from file
    int loadFile(string filename)
    {
        ifstream infile;
        open_file(infile,filename.c_str());
        if (!infile) {
            cerr<< "cannot open" <<endl;
            return 1;
        }
        
        // Load general parameters for a neural network
        infile >> numInputs >> numLayers >> epsilon;
        
        // Load objective function
        string temp, func_name;
        objFunctionMap<T> objFuncMap;
        infile >> func_name;
        objFunc = objFuncMap[func_name];
        
        // Clear existing memory to avoid size mismatch
        vector<hiddenLayer<T>>().swap(hiddenLayers);
        // Allocate memory for new settings to store information of hidden layers
        hiddenLayers.resize(numLayers);

        // Load detailed settings for a neural network, i.e. information of hidden layers
        trsFunctionMap<T> trsFuncMap;
        for (int l = 0; l < numLayers; l++) {
            infile >> temp;     // Jump line "{"
            // Load general parameters for a hidden layer
            int numLayerInputs, numNodes;
            string func_name = "";
            infile >> numLayerInputs >> numNodes >> func_name;
            hiddenLayers[l].numInputs = numLayerInputs;
            hiddenLayers[l].numNodes = numNodes;
            hiddenLayers[l].transFcn = trsFuncMap[func_name][0];
            hiddenLayers[l].transDrv = trsFuncMap[func_name][1];
            
            // Allocate memory for current layer to store information of nodes
            hiddenLayers[l].nodes.resize(numNodes);
            // Save detailed settings for a hidden layer, i.e. information of nodes
            for (int i = 0; i < numNodes; i++) {
                infile >> temp;     // Jump line "    ["
                // Save numInputs, bias and cum_bias of the node
                hiddenLayers[l].nodes[i].numInputs = numLayerInputs;
                infile >> hiddenLayers[l].nodes[i].bias >> hiddenLayers[l].nodes[i].cum_bias;
                // Allocate memory for current node to store information of weights and cum_weights
                hiddenLayers[l].nodes[i].weights.resize(numLayerInputs);
                hiddenLayers[l].nodes[i].cum_weights.resize(numLayerInputs);
                // Load weights of the node
                for (int k = 0; k < numLayerInputs; k++)
                    infile >> hiddenLayers[l].nodes[i].weights[k];
                // Load cum_weights of the node
                for (int k = 0; k < numLayerInputs; k++)
                    infile >> hiddenLayers[l].nodes[i].cum_weights[k];
                infile >> temp;  // Jump line "     ];"
            }
            infile >> temp;      // Jump line "};" and the following empty line
        }
        return 0;
    }

private:
	// Feed-forward algorithm to compute the output of network
	void feedForward(vector<batchData<T>>& inputs)
    {
		layerOutputs[0] = hiddenLayers[0].computeOutputs(inputs);
        for (int l = 1; l < numLayers; l++) {
			layerOutputs[l] = hiddenLayers[l].computeOutputs(layerOutputs[l-1]);
        }
    }

	// Back-propagation algorithm to update weights
	void backPropagation(vector<batchData<T>>& inputs, vector<batchData<T>>& targets)
    {
        // Initialize temporary variables
		size_t batchSize = inputs[0].size();
		vector<vector<batchData<T>>> delta(numLayers);
		vector<size_t> numNodes(numLayers);
		for (int l = 0; l < numLayers; l++){
			numNodes[l] = hiddenLayers[l].nodes.size();
			delta[l] = vector<batchData<T>>(numNodes[l],batchData<T>(batchSize));
		}
		vector<T> d_weights;
		T d_bias;
        
        // The last hidden layer depends on derivative of objective function
		vector<batchData<T>> prediction = layerOutputs[numLayers-1]; // B * N_L
        vector<batchData<T>> outputs = layerOutputs[numLayers-2]; // B * N_L
		vector<batchData<T>> outputs_drv = hiddenLayers[numLayers-1].getOutputs_drv();
#pragma omp parallel for private(d_weights, d_bias)
		for (int i = 0; i < numNodes[numLayers-1]; i++) {
			delta[numLayers-1][i] = objFunc.drv(targets[i], prediction[i]) * outputs_drv[i];
			d_weights = dot(delta[numLayers-1][i], outputs, true); // B * (B * N_L)
            d_bias = delta[numLayers-1][i].sum() / batchSize;
			hiddenLayers[numLayers-1].nodes[i].setDParams(d_weights,d_bias);
		}

        // The rest hidden layers depend on delta values of its successive layer
		for (int l = numLayers-2; l >= 0; l--) {
            outputs = (l==0) ? inputs : layerOutputs[l-1];
			outputs_drv = hiddenLayers[l].getOutputs_drv();
#pragma omp parallel for private(d_weights, d_bias)
			for (int i = 0; i < numNodes[l]; i++) {
				for (int j = 0; j < numNodes[l+1]; j++)
					for (int b = 0; b < batchSize; b++)
						delta[l][i][b] += delta[l+1][j][b] * hiddenLayers[l+1].nodes[j].weights[i];
				delta[l][i] = delta[l][i] * outputs_drv[i] / batchSize;
				d_weights = dot(delta[l][i], outputs, true);
                d_bias = delta[l][i].sum() / batchSize;
				hiddenLayers[l].nodes[i].setDParams(d_weights,d_bias);
			}
		}
		
        // Update all parameters after computing all gradients
		updateParameter();
	}
	
	// Update parameter
	void updateParameter()
	{
		for (int l = 0; l < numLayers; l++)
			hiddenLayers[l].updateParameter(epsilon);
	}
    
    // Divide data into batches, if number of data is not multiple of batch size, the remainder will be dropped
    vector<networkBatchInput<T>> divide(const vector<vector<T>>& Data, const vector<vector<T>>& Target, size_t batchSize, bool shuffle = true)
    {
        assert(batchSize > 0);
        assert(Data.size() == Target.size());
        
        vector<int> perm(Target.size(),-1);
        for (int i = 0; i < Target.size(); i++)
            perm[i] = i;
        // Shuffle the order
        if (shuffle)
            random_shuffle(perm.begin(), perm.end(), [](int n){ return rand() % n; });
        
        // Initialize the output
        size_t numBatch = Data.size()/batchSize;
        size_t inputDim = Data[0].size();           // Should be same as numInputs
        size_t targetDim = Target[0].size();
        
        // Construct batch input structure vector
        vector<networkBatchInput<T>> batchedInput(numBatch, networkBatchInput<T>(batchSize, inputDim, targetDim));
        for (int i = 0; i < numBatch; i++) {
            // Assign input values
            for (int j = 0; j < inputDim; j++)
                for (int b = 0; b < batchSize; b++)
                    batchedInput[i].input[j][b] = Data[perm[i*batchSize+b]][j];
            // Assign target values
            for (int j = 0; j < targetDim; j++)
                for (int b = 0; b < batchSize; b++)
                    batchedInput[i].target[j][b] = Target[perm[i*batchSize+b]][j];
        }
        return batchedInput;
    }
	
    
    /* File operation functions */
    
    // Open file for save settings
    ofstream& open_file(ofstream &outfile, const string &file)
    {
        outfile.close();
        outfile.clear();
        outfile.open(file.c_str());
        return outfile;
    }
    
    // Open file for load settings
    ifstream& open_file(ifstream &infile, const string &file)
    {
        infile.close();
        infile.clear();
        infile.open(file.c_str());
        return infile;
    }
};

#endif