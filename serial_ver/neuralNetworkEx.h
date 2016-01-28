#ifndef NEURALNETWORKEX_
#define NEURALNETWORKEX_

#include <cstdlib>
#include <algorithm>
#include <ctime>
#include "hiddenLayer.h"
#include "cmbFuncMap.h"
#include "objFuncMap.h"
#include "netInfo.h"
#include "RNG.h"

// Struct of network batch input for neuralNetworkEx class
template<typename T>
struct networkBatchInputEx {
    vector<vector<batchData<T>>> input;
    vector<batchData<T>> target;
    networkBatchInputEx(size_t batchSize, vector<size_t> inputDims, size_t targetDim) {
        size_t numSource = inputDims.size();
        input = vector<vector<batchData<T>>>(numSource);
        for (int s = 0; s <numSource; s++)
            input[s] = vector<batchData<T>>(inputDims[s], batchData<T>(batchSize));
        target = vector<batchData<T>>(targetDim, batchData<T>(batchSize));
    }
};

// Regard each hidden layer as a super node, the usual ones are chain structure.
// In this extension, the network is a DAG at the scale of super nodes.
// Also, there may be generalized transfer functon - "combine function" between a super node and all its parents
template <typename T>
class neuralNetworkEx {
private:
    netInfo netStruct;                              // All structure infomation of the neural network
    vector<vector<batchData<T>>> layerInputs;       // Use to store input of each hidden layer
    vector<vector<batchData<T>>> layerOutputs;      // Use to store output of each hidden layer
    vector<vector<batchData<T>>> layerOutputs_drv;  // Use to store output derivative of each hidden layer
    vector<cmbFuncInfo<T>> cmbFuncList;             // Combinine function and its partial derivatives in factor node
    objFuncInfo<T> objFunc;                         // Objective function and its partial derivative w.r.t. output
    vector<hiddenLayer<T>> hiddenLayers;            // array of hidden layers
    float epsilon;                                  // Unit step size for computing adaptive step size
public:
    // Constructor
    neuralNetworkEx(vector<int> inputDims, vector<string> inputDists, vector<int> layerInputDims, vector<int> numNodesPerLayer, vector<string> trsFuncName, vector<string> cmbFuncName, vector<vector<int>> I2L_edges, vector<vector<int>> L2L_edges, string objFuncName, float _epsilon = 0.01) : epsilon(_epsilon) {
        size_t numLayers = numNodesPerLayer.size();
        assert(inputDims.size() > 0 && numLayers > 0);
        assert( numLayers == trsFuncName.size() && numLayers == cmbFuncName.size() );
        // Construct DAG structure of the neural network
        netStruct = netInfo(inputDims, inputDists, layerInputDims, numNodesPerLayer, trsFuncName, cmbFuncName, I2L_edges, L2L_edges);
        // Malloc memory for detailed information of hidden layers
        trsFunctionMap<T> trsFuncMap;
        cmbFunctionMap<T> cmbFuncMap;
        objFunctionMap<T> objFuncMap;
        hiddenLayers.resize(numLayers);
        cmbFuncList.resize(numLayers);
        for (int i = 0; i < numLayers; i++) {
            layerInfoStr info = netStruct.layerInfo[i];
            hiddenLayers[i] = hiddenLayer<T>(info.inputDim, info.numNodes, trsFuncMap[info.trsFuncName]);
            cmbFuncList[i] = cmbFuncMap[info.cmbFuncName];
        }
        objFunc = objFuncMap[objFuncName];
        // Malloc memory for layer input and output
        layerInputs.resize(numLayers);
        layerOutputs.resize(numLayers);
        layerOutputs_drv.resize(numLayers);
//        netStruct.print();  // Print netInfo for debug
    }
    
    
    // Constructor from file
    neuralNetworkEx(string filename)
    {
        this->loadFile(filename);
        // Malloc memory for layer input and output
        layerInputs.resize(netStruct.numLayers);
        layerOutputs.resize(netStruct.numLayers);
        layerOutputs_drv.resize(netStruct.numLayers);
    }
    
    // Destructor
    virtual ~neuralNetworkEx()
    {
        vector<hiddenLayer<T>>().swap(hiddenLayers);
    }
    
    // Assign weights for all edges in the network
    void setParams(vector<vector<vector<T>>> weights, vector<vector<T>> bias)
    {
        for (int i = 0; i < netStruct.numLayers; i++)
            hiddenLayers[i].setWeights(weights[i], bias[i]);
    }
    
    // Train the neural network and store training/testing MSE for each epoch
    vector<vector<T>> train(int epoch, int batchSize, const vector<vector<vector<T>>>& trainData, const vector<vector<T>>& trainTarget, const vector<vector<vector<T>>>& testData, const vector<vector<T>>& testTarget, bool print_info = true)
    {
        // Guarantee that the number of input sources are same for train and test
        assert( trainData.size() > 0 && testData.size() == trainData.size() );
        // Guarantee that the sample number for input and target are same
        for (int s = 0; s < trainData.size(); s++)
            assert( trainData[s].size() == trainTarget.size() );
        for (int s = 0; s < testData.size(); s++)
            assert( testData[s].size() == testTarget.size() );
        
        size_t numTrainSample = trainTarget.size();
        size_t numTrainBatch = numTrainSample / batchSize;
        size_t numTestSample = testTarget.size();
        size_t numTestBatch = numTestSample / batchSize;
        
        // Initialize vector to store loss for each epoch
        vector<T> trainLoss(epoch,0);
        vector<T> testLoss(epoch,0);
        
        // Divide testing data into batches: batch number index, input index, batchData index
        if (print_info)
            cout << "Permute testing samples and divide them into batches" << endl;
        vector<networkBatchInputEx<T>> testBatches = divide(testData, testTarget, batchSize);

        vector<batchData<T>> outputs;
        T temp = 0;
        clock_t timer, time_elapse;
        for (int t = 0; t < epoch; t++) {
            // Randomly permute training data for stochastic gradient descent algorithm
            if (print_info)
                cout << "Permute training samples and divide them into batches" << endl;
            vector<networkBatchInputEx<T>> trainBatches = divide(trainData, trainTarget, batchSize);
            // Train the network using training data and keep track of training error
            for (int i = 0; i < numTrainBatch; i++) {
	            timer = clock();
                feedForward(trainBatches[i].input);
                backPropagation(trainBatches[i].input, trainBatches[i].target);
                outputs = layerOutputs[netStruct.numLayers-1];
                temp = objFunc.func(trainBatches[i].target, outputs);
                trainLoss[t] += temp;
                
                time_elapse = clock() - timer;
                if (print_info)
                    cout << "Epoch " << t << ", batch " << i << ", train current batch loss = " << temp / batchSize << ", used " << (float)time_elapse / CLOCKS_PER_SEC << " sec" << endl;
            }
            // Test the network using testing data and keep track of testing error
            for (int i = 0; i < numTestBatch; i++) {
                timer = clock();
                feedForward(testBatches[i].input);
                outputs = layerOutputs[netStruct.numLayers-1];
                temp = objFunc.func(testBatches[i].target, outputs);
                testLoss[t] += temp;
                
                time_elapse = clock() - timer;
                if (print_info)
                    cout << "Epoch " << t << ", batch " << i << ", test current batch loss = " << temp / batchSize << ", used " << (float)time_elapse / CLOCKS_PER_SEC << " sec" << endl;
            }
            if (print_info)
                cout << "train loss = " << trainLoss[t] / trainData.size() << ",\t test loss = " << testLoss[t] / testData.size() << endl;
        }
        return {trainLoss, testLoss};
    }

    // Train the neural network and store training MSE, do not use testing data during training procedure
    vector<T> train(int epoch, int batchSize, const vector<vector<vector<T>>>& trainData, const vector<vector<T>>& trainTarget, bool print_info = true)
    {
        // Guarantee that the sample number for input and target are same
        for (int s = 0; s < trainData.size(); s++)
            assert( trainData[s].size() == trainTarget.size() );
        
        size_t numTrainSample = trainTarget.size();
        size_t numTrainBatch = numTrainSample / batchSize;
        
        // Initialize vector to store loss for each epoch
        vector<T> trainLoss(epoch,0);
        
        vector<batchData<T>> outputs;
        T temp = 0;
        clock_t timer, time_elapse;
        for (int t = 0; t < epoch; t++) {
            // Randomly permute training data for stochastic gradient descent algorithm
            if (print_info)
                cout << "Permute training samples and divide them into batches" << endl;
            vector<networkBatchInputEx<T>> trainBatches = divide(trainData, trainTarget, batchSize);
            timer = clock();
            // Train the network using training data and keep track of training error
            for (int i = 0; i < numTrainBatch; i++) {
                timer = clock();
                feedForward(trainBatches[i].input);
                backPropagation(trainBatches[i].input, trainBatches[i].target);
                outputs = layerOutputs[netStruct.numLayers-1];
                temp = objFunc.func(trainBatches[i].target, outputs);
                trainLoss[t] += temp;
                
                time_elapse = clock() - timer;
                if (print_info)
                    cout << "Epoch " << t << ", batch " << i << ", train current batch loss = " << temp / batchSize << ", used " << (float)time_elapse / CLOCKS_PER_SEC << " sec" << endl;
            }
            if (print_info)
                cout  << "Epoch " << t << ", train loss = " << trainLoss[t] / trainData.size() << endl;
        }
        return trainLoss;
    }
    
    // Given input data, use the neural network to generate the output
    vector<vector<T>> apply(const vector<vector<vector<T>>>& inputData, int batchSize = 100, bool print_info = true)
    {
        assert( inputData.size() > 0 );
        
        size_t numSample = inputData[0].size();
        size_t numBatch = numSample / batchSize;
        
        // Divide testing data into batches without permute
        vector<vector<T>> dummy_inputTarget(numSample);
        vector<networkBatchInputEx<T>> batches = divide(inputData, dummy_inputTarget, batchSize, false);
        if (print_info)
            cout << "Divided input data into " << numBatch << " batches" << endl;
        
        // Pass the input through network to get output
        clock_t timer, time_elapse;
        vector<vector<batchData<T>>> outputs(numBatch);
        for (int i = 0; i < numBatch; i++) {
            timer = clock();
            feedForward(batches[i].input);
            outputs[i] = layerOutputs[netStruct.numLayers-1];
            time_elapse = clock() - timer;
            if (print_info)
                cout << "Batch " << i << " out of " << numBatch << " finished computation" << ", used " << (float)time_elapse / CLOCKS_PER_SEC << " sec" << endl;
        }
        
        // Convert vector<vector<batchData<T>>> back to vector<vector<T>>
        size_t outputDim = layerOutputs[netStruct.numLayers-1].size();
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
            cerr << "cannot open" << endl;
            return 1;
        }
        
        // Save epsilon
        outfile << epsilon << endl << endl;
        
        // Save netInfo structure
        outfile << netStruct << endl;
        
        // Save objective function name
        objFunctionMap<T> objFuncMap;
        outfile << objFuncMap(objFunc.func) << endl;

        // Save detailed settings for a neural network, i.e. information of hidden layers
        trsFunctionMap<T> trsFuncMap;
        for (int l = 0; l < netStruct.numLayers; l++) {
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
        // Empty previous netInfo structure
        netStruct.clear();
        
        ifstream infile;
        open_file(infile,filename.c_str());
        if (!infile) {
            cerr << "cannot open" << endl;
            return 1;
        }
        
        string temp, func_name;

        // Load epsilon and netInfo structure
        infile >> epsilon;
        infile >> netStruct;
        
        // Load objective function
        objFunctionMap<T> objFuncMap;
        infile >> func_name;
        objFunc = objFuncMap[func_name];
        
        // Load combine functions according to netInfo
        cmbFunctionMap<T> cmbFuncMap;
        for (int l = 0; l < netStruct.numLayers; l++) {
            cmbFuncList[l] = cmbFuncMap[netStruct.layerInfo[l].cmbFuncName];
        }
                    
        // Clear existing memory to avoid size mismatch
        vector<hiddenLayer<T>>().swap(hiddenLayers);
        // Allocate memory for new settings to store information of hidden layers
        hiddenLayers.resize(netStruct.numLayers);
        
        // Load detailed settings for a neural network, i.e. information of hidden layers
        trsFunctionMap<T> trsFuncMap;
        for (int l = 0; l < netStruct.numLayers; l++) {
            infile >> temp;     // Jump line "{"
            // Load general parameters for a hidden layer
            int numLayerInputs, numNodes;
            infile >> numLayerInputs >> numNodes >> func_name;
            hiddenLayers[l].numInputs = numLayerInputs;
            hiddenLayers[l].numNodes = numNodes;
            hiddenLayers[l].transFcn = trsFuncMap[func_name][0];
            hiddenLayers[l].transDrv = trsFuncMap[func_name][1];
            
            // Allocate memory for current layer to store information of nodes
            hiddenLayers[l].nodes.resize(numNodes);
            // Save detailed settings for a hidden layer, i.e. information of nodes
            for (int i = 0; i < numNodes; i++) {
                infile >> temp;  // Jump line "    ["
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
                infile >> temp;  // Jump line "    ];"
            }
            infile >> temp;      // Jump line "};" and the following empty line
        }
        return 0;
    }
    
private:
    // Feed-forward algorithm to compute the output of network
    void feedForward(vector<vector<batchData<T>>>& inputs)
    {
        static bool flag = false;   // Only compute input pointer for combine function once
        // For each stage of forward scheduler
        for (int stage = 0; stage < netStruct.schedulerFwd.size(); stage++ )
            // For each layer in the current stage
            for (int l = 0; l < netStruct.schedulerFwd[stage].size(); l++) {
                // Get the layer index
                int layerIdx = netStruct.schedulerFwd[stage][l];
                /* Compute the input of current layer */
                // If the combine function is not trivial, i.e. not identity function
                // Combine all outputs from parent layers and connected input source to compute input of the current layer
                if (cmbFuncList[layerIdx].func != NULL) {
                    // Get combine function of current layer
                    cmb_func_type<batchData<T>> combinefunc = cmbFuncList[layerIdx].func;
                    if (!flag) {
                        flag = true;
	                    // Clear pervious inputs pointers
	                    cmbFuncList[layerIdx].inputPtr.clear();
	                    // If current layer is connected to an input source, store its position in the combine function
	                    int inputPos = (netStruct.srcList[layerIdx][0] == -1) ? -1 : netStruct.srcList[layerIdx][1];
	                    int numParam = (int) (inputPos != -1) + (int) netStruct.parentList[layerIdx].size();
	                    // Construct vector of input pointers for combine function
	                    for (int j = 0, p = 0; j < numParam; j++) {
	                        if (j == inputPos)
	                            cmbFuncList[layerIdx].inputPtr.push_back(&inputs[netStruct.srcList[layerIdx][0]]);
	                        else {
	                            // Get the parent layer index of the p-th parent of current layer index
	                            int parent_layerIdx = netStruct.parentList[layerIdx][p];
	                            cmbFuncList[layerIdx].inputPtr.push_back(&layerOutputs[parent_layerIdx]);
	                            p++;
                            }
                        }
                    }
                    // Given the vector of all input pointers, compute output of combine function as input of current layer
                    layerInputs[layerIdx] = combinefunc(cmbFuncList[layerIdx].inputPtr);
                }
                // If the combine function is trivial, i.e. identity function
                // Then it mean this layer is only connected to one input source, or one parent layer
                else {
                    // If current layer is only connected to an input source
                    // Then the input of current layer is this input source
                    if (netStruct.srcList[layerIdx][0] != -1)
                        layerInputs[layerIdx] = inputs[netStruct.srcList[layerIdx][0]];
                    // If current layer is only connected to a parent layer
                    // Then the input of current layer is the output of its unique parent layer
                    else
                        layerInputs[layerIdx] = layerOutputs[netStruct.parentList[layerIdx][0]];
                }
                /* Compute the output and its derivative of current layer */
                layerOutputs[layerIdx] = hiddenLayers[layerIdx].computeOutputs(layerInputs[layerIdx]);
                layerOutputs_drv[layerIdx] = hiddenLayers[layerIdx].getOutputs_drv();
            }
    }

    // Back-propagation algorithm to update weights
    void backPropagation(vector<vector<batchData<T>>>& inputs, vector<batchData<T>>& targets)
    {
        // Initialize temporary variables
        size_t batchSize = inputs[0][0].size();
        vector<vector<batchData<T>>> delta(netStruct.numLayers);
        vector<size_t> numNodes(netStruct.numLayers);
        int numLayers = netStruct.numLayers;
        for (int l = 0; l < numLayers; l++){
            numNodes[l] = netStruct.layerInfo[l].numNodes;
            delta[l] = vector<batchData<T>>(numNodes[l],batchData<T>(batchSize));
        }
        vector<T> d_weights;
        T d_bias;
        vector<batchData<T>> outputs_drv;
        
        /* The last hidden layer depends on derivative of objective function w.r.t. the output */
        // Final prediction of the neural network, i.e. the output of last hidden layer
        vector<batchData<T>> prediction = layerOutputs[numLayers-1];
        // Derivative of output w.r.t. linear output, i.e. derivative of transfer function
        outputs_drv = layerOutputs_drv[numLayers-1];
        
        for (int i = 0; i < numNodes[numLayers-1]; i++) {
            delta[numLayers-1][i] = objFunc.drv(targets[i], prediction[i]) * outputs_drv[i] / batchSize;
            d_weights = dot(delta[numLayers-1][i], layerInputs[numLayers-1]);
            d_bias = delta[numLayers-1][i].sum();
            hiddenLayers[numLayers-1].nodes[i].setDParams(d_weights,d_bias);
        }
        
        /* The rest hidden layers depend on derivatives of combine functions w.r.t. their child */
        vector<batchData<T>> cmbFunDrv_output;
        // For each stage of backward scheduler
        for (int stage = 1; stage < netStruct.schedulerBk.size(); stage++)
            // For each layer in the current stage
            for (int l = 0; l < netStruct.schedulerBk[stage].size(); l++) {
                // Get the layer index
                int layerIdx = netStruct.schedulerBk[stage][l];
                /* Compute detla for current layer */
                // For each child layer of current layer, compute their delta and sum them up
                for (int c = 0; c < netStruct.childList[layerIdx].size(); c++) {
                    // Get the child layer index
                    int child_layerIdx = netStruct.childList[layerIdx][c];
                    // Compute the position of element 'layerIdx' in the vector 'parentList[child_layerIdx]'
                    int pc = netStruct.getParentListIndex(child_layerIdx,layerIdx);
                    // If there is a non-trival combine function between layers 'layerIdx' and 'child_layerIdx'
                    // The partial derivative w.r.t. output of layer 'layerIdx' need to be included
                    if (cmbFuncList[child_layerIdx].func != NULL) {
                        // Compute the partial derivative of the combine function of layer 'child_layerIdx' w.r.t. layer 'layerIdx'
                        cmb_func_type<batchData<T>> cmbFunc_drv = cmbFuncList[child_layerIdx].drv[pc];
                        cmbFunDrv_output = cmbFunc_drv(cmbFuncList[child_layerIdx].inputPtr);
                        for (int i = 0; i < numNodes[layerIdx]; i++)
                            for (int j = 0; j < numNodes[child_layerIdx]; j++)
                                for (int b = 0; b < batchSize; b++)
                                    delta[layerIdx][i][b] += delta[child_layerIdx][j][b] * hiddenLayers[child_layerIdx].nodes[j].weights[i] * cmbFunDrv_output[i][b];
                    }
                    // If there is a trival combine function between layers 'layerIdx' and 'child_layerIdx'
                    // The partial derivative w.r.t. output of layer 'layerIdx' is just one and can be omitted
                    else
                        for (int i = 0; i < numNodes[layerIdx]; i++)
                            for (int j = 0; j < numNodes[child_layerIdx]; j++)
                                for (int b = 0; b < batchSize; b++)
                                    delta[layerIdx][i][b] += delta[child_layerIdx][j][b] * hiddenLayers[child_layerIdx].nodes[j].weights[i];
                }
                /* Compute d_W, d_b for current layer */
                // Derivative of output w.r.t. linear output, i.e. derivative of transfer function
                outputs_drv = hiddenLayers[layerIdx].getOutputs_drv();
                for (int i = 0; i < numNodes[layerIdx]; i++) {
                    delta[layerIdx][i] = delta[layerIdx][i] * outputs_drv[i] / batchSize;
                    d_weights = dot(delta[layerIdx][i], layerInputs[layerIdx]);
                    d_bias = delta[layerIdx][i].sum();
                    hiddenLayers[layerIdx].nodes[i].setDParams(d_weights,d_bias);
                }
            }
        
        // Update all parameters after computing all gradients
        updateParameter();
    }
    
    // Update parameter
    void updateParameter()
    {
        for (int l = 0; l < netStruct.numLayers; l++)
            hiddenLayers[l].updateParameter(epsilon);
    }

    // Divide data into batches, if number of data is not multiple of batch size, the remainder will be dropped
    vector<networkBatchInputEx<T>> divide(const vector<vector<vector<T>>>& Data, const vector<vector<T>>& Target, size_t batchSize, bool shuffle = true)
    {
        assert(batchSize > 0);
        // Guarantee the sample numbers of all inputs sources are same
        // If the sample vector is empty, then it is a random input
        vector<bool> isRandom(Data.size());
        for (int s = 1; s < Data.size(); s++) {
            assert(Data[s].size() > 0 && Data[s].size() == Target.size());
            isRandom[s] = (Data[s][0].size() == 0);
        }
        
        vector<int> perm(Target.size(),-1);
        for (int i = 0; i < Target.size(); i++)
            perm[i] = i;
        // Shuffle the order
        if (shuffle)
            random_shuffle(perm.begin(), perm.end(), [](int n){ return rand() % n; });

        // Initialize the output
        size_t numSource = Data.size();
        size_t numSample = Target.size();
        size_t numBatch = numSample / batchSize;
        vector<size_t> inputDims(numSource);
        for (int s = 0; s < numSource; s++)
            inputDims[s] = netStruct.srcDim[s];
        size_t targetDim = Target[0].size();
        
        // Construct batch input structure vector
        vector<networkBatchInputEx<T>> batchedInput(numBatch, networkBatchInputEx<T>(batchSize, inputDims, targetDim));
        for (int i = 0; i < numBatch; i++) {
            // Assign inputs values
            for (int s = 0; s < numSource; s++) {
                // For fixed input data, permute the samples and batch them
                if (!isRandom[s]) {
                    for (int j = 0; j < inputDims[s]; j++)
                        for (int b = 0; b < batchSize; b++)
                            batchedInput[i].input[s][j][b] = Data[s][perm[i*batchSize+b]][j];
                }
                // For random input data, generate them according to their distribution
                else
                    batchedInput[i].input[s] = randomGenerator<T>(netStruct.srcDist[s], inputDims[s], batchSize);
            }
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