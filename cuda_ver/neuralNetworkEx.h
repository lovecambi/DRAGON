#ifndef NEURALNETWORKEX_
#define NEURALNETWORKEX_

#include <cstdlib>
#include <algorithm>
#include <time.h>
#include "hiddenLayer.h"
#include "cmbFuncMap.h"
#include "objFuncMap.h"
#include "netInfo.h"
#include "RNG.h"

// Struct of network batch input for neuralNetworkEx class
struct networkBatchInputEx_host {
    vector<thrust::host_vector<float>> input;
    thrust::host_vector<float> target;
    networkBatchInputEx_host(size_t batchSize = 1, vector<size_t> inputDims = {1}, size_t targetDim = 1) 
    {
        for (int s = 0; s < inputDims.size(); s++)
            input.push_back(thrust::host_vector<float>(inputDims[s]*batchSize));
        target = thrust::host_vector<float>(targetDim*batchSize);
    }
};

struct networkBatchInputEx_device {
    vector<thrust::device_vector<float>> input;
    thrust::device_vector<float> target;
    networkBatchInputEx_device(size_t batchSize = 1, vector<size_t> inputDims = {1}, size_t targetDim = 1) 
    {
        for (int s = 0; s < inputDims.size(); s++)
            input.push_back(thrust::host_vector<float>(inputDims[s]*batchSize));
        target = thrust::device_vector<float>(targetDim*batchSize);
    }
    void copy(networkBatchInputEx_host& host_batch) 
    {
        if (input.size() != host_batch.input.size())
            input.resize(host_batch.input.size());
        for (int s = 0; s < host_batch.input.size(); s++)
            input[s] = host_batch.input[s];
        target = host_batch.target;
    }
};

// Regard each hidden layer as a super node, the usual ones are chain structure.
// In this extension, the network is a DAG at the scale of super nodes.
// Also, there may be generalized transfer functon - "combine function" between a super node and all its parents
class neuralNetworkEx {
protected:
    netInfo netStruct;                                          // All structure infomation of the neural network
    bool lastLayerAsOutput;                                     // Is true if last layer only combine its parent layers output and has no weight and bias, 
                                                                // otherwise is bias. W = identity, b = 0, these are not changed in training
    int batchSize;                                              // batch size of batch training
    cublasHandle_t handle;                                      // cuBlas handle for matrix operation
    vector<cmbFuncInfo> cmbFuncList;                            // Combinine function and its partial derivatives in factor node
    objFuncInfo objFunc;                                        // Objective function and its partial derivative w.r.t. output
    float epsilon;                                              // Unit step size for computing adaptive step size
    
    vector<hiddenLayer> hiddenLayers;                           // array of hidden layers
    vector<thrust::device_vector<float>> layerInputs;           // Use to store input of each hidden layer
    vector<thrust::device_vector<float>> layerOutputs;          // Use to store output of each hidden layer
    vector<thrust::device_vector<float>> layerOutputs_drv;      // Use to store output derivative of each hidden layer
    vector<vector<thrust::device_vector<float>>> cmbOutputs_drv;// Use to store output derivative of combine function of each hidden layer w.r.t. each parent layer output

    vector<vector<thrust::device_vector<float>>> delta;         // Partial derivative of objective function w.r.t. linear outputs, each part comes from one child layer
    vector<thrust::device_vector<float>> weights_drv;           // Partial derivative of objective function w.r.t. weights
    vector<thrust::device_vector<float>> bias_drv;              // Partial derivative of objective function w.r.t. bias
    thrust::device_vector<float> one_vec;                       // auxiliary vector of all ones

    RNG random_generator;                                       // Use cuRand to generate random variable
    vector<thrust::device_vector<float>> temp_rand_input;       // Use for generate random input values
public:
    // Constructor
    neuralNetworkEx(vector<int> inputDims = vector<int>(1), 
                    vector<string> inputDists = vector<string>(1), 
                    vector<int> layerInputDims = vector<int>(1), 
                    vector<int> numNodesPerLayer = vector<int>(1), 
                    int _batchSize = 1,
                    vector<string> trsFuncName = vector<string>(1), 
                    vector<string> cmbFuncName = vector<string>(1), 
                    vector<vector<int>> I2L_edges = vector<vector<int>>(), 
                    vector<vector<int>> L2L_edges = vector<vector<int>>(), 
                    string objFuncName = "", 
                    float _epsilon = 0.01,
                    int _lastLayerAsOutput = 0) : batchSize(_batchSize), epsilon(_epsilon), lastLayerAsOutput(_lastLayerAsOutput)
    {
        size_t numLayers = numNodesPerLayer.size();
        assert( inputDims.size() > 0 && numLayers > 0 );
        assert( numLayers == trsFuncName.size() && numLayers == cmbFuncName.size() );
        // Construct DAG structure of the neural network
        netStruct = netInfo(inputDims, inputDists, layerInputDims, numNodesPerLayer, trsFuncName, cmbFuncName, I2L_edges, L2L_edges);
        // Malloc memory for detailed information of hidden layers
        trsFunctionMap trsFuncMap;
        cmbFunctionMap cmbFuncMap;
        hiddenLayers.resize(numLayers);
        hiddenLayers.resize(numLayers);
        for (int i = 0; i < numLayers; i++) {
            layerInfoStr info = netStruct.layerInfo[i];
            hiddenLayers[i] = hiddenLayer(info.inputDim, info.numNodes, batchSize, trsFuncMap[info.trsFuncName]);
            cmbFuncList.push_back(cmbFuncMap[info.cmbFuncName]);
        }
        objFunctionMap objFuncMap;
        objFunc = objFuncMap[objFuncName];
        // Create a handle for CUBLAS
        cublasCreate(&handle);
        // Malloc memory for layer input and output
        initialize();
        // If last layer serves as output layer, set W = identity, b = 0
        if (lastLayerAsOutput) {
            layerInfoStr info = netStruct.layerInfo[numLayers-1];
            // The input dimension and output dimension of output layer must be same
            assert( info.inputDim == info.numNodes );
            // Set W = identity for output layer
            thrust::device_vector<float> identity_matrix(info.numNodes*info.inputDim);
            for (int i = 0; i < info.numNodes; i++)
                identity_matrix[i*info.inputDim+i] = 1;
            thrust::copy(identity_matrix.begin(), identity_matrix.end(), hiddenLayers[numLayers-1].weights.begin());
        }
    }
    
    
    // Constructor from file
    neuralNetworkEx(string filename)
    {
        this->loadFile(filename);
        // Create a handle for CUBLAS
        cublasCreate(&handle);
        // Malloc memory for layer input and output
        initialize();
    }
    
    // Destructor
    virtual ~neuralNetworkEx()
    {
        // Destroy the handle
        cublasDestroy(handle);
        // device_vector are automatically released by cudaFree()
    }
    
    // Assign weights for all edges in the network
    void setParams(vector<thrust::device_vector<float>>& weights, vector<thrust::device_vector<float>> bias)
    {
        for (int i = 0; i < netStruct.numLayers; i++)
            hiddenLayers[i].setParams(weights[i], bias[i]);
    }
    
    // Train the neural network and store training/testing MSE for each epoch
    vector<vector<float>> train(int epoch, const vector<vector<vector<float>>>& trainData, const vector<vector<float>>& trainTarget, const vector<vector<vector<float>>>& testData, const vector<vector<float>>& testTarget, bool print_info = true)
    {
        // Guarantee that the number of input sources are same for train and test
        assert( trainData.size() == netStruct.numSource && testData.size() == trainData.size() );
        // Guarantee that the sample number for input and target are same
        for (int s = 0; s < trainData.size(); s++)
            assert( trainData[s].size() == trainTarget.size() );
        for (int s = 0; s < testData.size(); s++)
            assert( testData[s].size() == testTarget.size() );
        
        size_t numTrainSample = trainTarget.size();
        size_t numTrainBatch = numTrainSample / batchSize;
        size_t numTestSample = testTarget.size();
        size_t numTestBatch = numTestSample / batchSize;
        
        vector<size_t> inputDim(netStruct.numSource);
        for (int s = 0; s < netStruct.numSource; s++)
            inputDim[s] = netStruct.srcDim[s];
        size_t targetDim = trainTarget[0].size();

        // Initialize vector to store loss for each epoch
        vector<float> trainLoss(epoch,0);
        vector<float> testLoss(epoch,0);
        
        // Divide testing data into batches: batch number index, input index, batchData index
        if (print_info)
            cout << "Permute testing samples and divide them into batches" << endl;
        vector<networkBatchInputEx_host> trainBatchesHost(numTrainBatch, networkBatchInputEx_host(batchSize, inputDim, targetDim));
        vector<networkBatchInputEx_host> testBatchesHost(numTestBatch, networkBatchInputEx_host(batchSize, inputDim, targetDim));
        networkBatchInputEx_device trainBatchDevice(batchSize, inputDim, targetDim);
        networkBatchInputEx_device testBatchDevice(batchSize, inputDim, targetDim);
        divide(testData, testTarget, testBatchesHost, false);

        thrust::device_vector<float> outputs;
        float temp = 0;
        clock_t timer, time_elapse;
        for (int t = 0; t < epoch; t++) {
            // Randomly permute training data for stochastic gradient descent algorithm
            if (print_info)
                cout << "Permute training samples and divide them into batches" << endl;
            divide(trainData, trainTarget, trainBatchesHost);
            // Train the network using training data
            timer = clock();
            for (int i = 0; i < numTrainBatch; i++) {
                trainBatchDevice.copy(trainBatchesHost[i]);
                feedForward(trainBatchDevice.input);
                backPropagation(trainBatchDevice.input, trainBatchDevice.target);

                time_elapse = clock() - timer;
                timer = clock();
                if (print_info)
                    cout << "Epoch " << t << ", batch " << i << " training used " << (float)time_elapse / CLOCKS_PER_SEC << " sec" << endl;
            
            }
            // Train the network using training data and keep track of training error
            for (int i = 0; i < numTrainBatch; i++) {
                trainBatchDevice.copy(trainBatchesHost[i]);
                feedForward(trainBatchDevice.input);
                outputs = layerOutputs[netStruct.numLayers-1];
                temp = thrust::inner_product(trainBatchDevice.target.begin(), trainBatchDevice.target.end(), 
                    outputs.begin(), 0.0f, thrust::plus<float>(), objFunc.func);
                trainLoss[t] += temp / numTrainSample;
                
                time_elapse = clock() - timer;
                timer = clock();
                if (print_info)
                    cout << "Epoch " << t << ", batch " << i << ", train current batch loss = " << temp / batchSize << ", used " << (float)time_elapse / CLOCKS_PER_SEC << " sec" << endl;
            }
            // Test the network using testing data and keep track of testing error
            for (int i = 0; i < numTestBatch; i++) {
                testBatchDevice.copy(testBatchesHost[i]);
                feedForward(testBatchDevice.input);
                outputs = layerOutputs[netStruct.numLayers-1];
                temp = thrust::inner_product(testBatchDevice.target.begin(), testBatchDevice.target.end(), 
                    outputs.begin(), 0.0f, thrust::plus<float>(), objFunc.func);
                testLoss[t] += temp / numTestSample;
                
                time_elapse = clock() - timer;
                timer = clock();
                if (print_info)
                    cout << "Epoch " << t << ", batch " << i << ", test current batch loss = " << temp / batchSize << ", used " << (float)time_elapse / CLOCKS_PER_SEC << " sec" << endl;
            }
            cout << "Epoch " << t << ", train loss = " << trainLoss[t] << ",\t test loss = " << testLoss[t] << endl;
        }
        return {trainLoss, testLoss};
    }

    // Train the neural network and store training MSE, do not use testing data during training procedure
    vector<float> train(int epoch, const vector<vector<vector<float>>>& trainData, const vector<vector<float>>& trainTarget, bool print_info = true)
    {
        // Guarantee that the sample number for input and target are same
        for (int s = 0; s < trainData.size(); s++)
            assert( trainData[s].size() == trainTarget.size() );
        
        size_t numTrainSample = trainTarget.size();
        size_t numTrainBatch = numTrainSample / batchSize;

        vector<size_t> inputDim(trainData.size());
        for (int s = 0; s < trainData.size(); s++)
            inputDim[s] = netStruct.srcDim[s];
        size_t targetDim = trainTarget[0].size();
        
        // Initialize vector to store loss for each epoch
        vector<float> trainLoss(epoch,0);

        vector<networkBatchInputEx_host> trainBatchesHost(numTrainBatch, networkBatchInputEx_host(batchSize, inputDim, targetDim));
        networkBatchInputEx_device trainBatchDevice(batchSize, inputDim, targetDim);
        
        thrust::device_vector<float> outputs;
        float temp = 0;
        clock_t timer, time_elapse;
        for (int t = 0; t < epoch; t++) {
            // Randomly permute training data for stochastic gradient descent algorithm
            if (print_info)
                cout << "Permute training samples and divide them into batches" << endl;
            divide(trainData, trainTarget, trainBatchesHost);
            timer = clock();
            // Train the network using training data and keep track of training error
            for (int i = 0; i < numTrainBatch; i++) {
                trainBatchDevice.copy(trainBatchesHost[i]);
                feedForward(trainBatchDevice.input);
                backPropagation(trainBatchDevice.input, trainBatchDevice.target);
                outputs = layerOutputs[netStruct.numLayers-1];
                temp = thrust::inner_product(trainBatchDevice.target.begin(), trainBatchDevice.target.end(), 
                    outputs.begin(), 0.0f, thrust::plus<float>(), objFunc.func);
                trainLoss[t] += temp / numTrainSample;
                
                time_elapse = clock() - timer;
                timer = clock();
                if (print_info)
                    cout << "Epoch " << t << ", batch " << i << ", train current batch loss = " << temp / batchSize << ", used " << (float)time_elapse / CLOCKS_PER_SEC << " sec" << endl;
            }
            cout  << "Epoch " << t << ", running train loss = " << trainLoss[t] << endl;
        }
        return trainLoss;
    }
    
    // Given input data, use the neural network to generate the output
    vector<vector<float>> apply(const vector<vector<vector<float>>>& inputData, bool print_info = true)
    {
        assert( inputData.size() > 0 );
        
        size_t numSample = inputData[0].size();
        size_t numBatch = numSample / batchSize;

        vector<size_t> inputDim(inputData.size());
        for (int s = 0; s < inputData.size(); s++)
            inputDim[s] = netStruct.srcDim[s];
        size_t targetDim = 0;
        size_t outputDim = hiddenLayers[netStruct.numLayers-1].numNodes;
        
        // Divide testing data into batches without permute
        vector<vector<float>> dummy_inputTarget(numSample);
        vector<networkBatchInputEx_host> batchesHost(numBatch, networkBatchInputEx_host(batchSize, inputDim, targetDim));
        networkBatchInputEx_device batchDevice(batchSize, inputDim, targetDim);
        divide(inputData, dummy_inputTarget, batchesHost, false);
        if (print_info)
            cout << "Divided input data into " << numBatch << " batches" << endl;
        
        // Pass the input through network to get output
        clock_t timer = clock(), time_elapse;
        vector<thrust::host_vector<float>> outputs(numBatch, vector<float>(outputDim*batchSize));
        for (int i = 0; i < numBatch; i++) {
            batchDevice.copy(batchesHost[i]);
            feedForward(batchDevice.input);
            thrust::copy(layerOutputs[netStruct.numLayers-1].begin(), layerOutputs[netStruct.numLayers-1].end(), outputs[i].begin());
            
            time_elapse = clock() - timer;
            timer = clock();
            if (print_info)
                cout << "Batch " << i << " out of " << numBatch << " finished computation" << ", used " << (float)time_elapse / CLOCKS_PER_SEC << " sec" << endl;
        }
        
        // Convert vector<vector<batchData<T>>> back to vector<vector<T>>
        vector<vector<float>> outputData(numSample, vector<float>(outputDim));
        for (int i = 0; i < numBatch; i++)
            for (int b = 0; b < batchSize; b++)
                thrust::copy(outputs[i].begin() + b*outputDim, outputs[i].begin() + (b+1)*outputDim, outputData[i*batchSize+b].begin());
        if (print_info)
            cout << "Finished conversion from batches to 2D-array" << endl;
        return outputData;
        // return vector<vector<float>>(3,vector<float>(5,1));
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
        outfile << (*this);
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
        infile >> (*this);
        infile.close();
        return 0;
    }

    // << overload: print the vector for debug
    friend ostream& operator<<(ostream& outfile, neuralNetworkEx& nn)
    {
        // Save epsilon
        outfile << nn.epsilon << " " << nn.batchSize << " " << endl << endl;
        // Save netInfo structure
        outfile << nn.netStruct << endl;
        // Save objective function name
        objFunctionMap objFuncMap;
        outfile << objFuncMap(nn.objFunc.func) << endl;
        // Save detailed settings for a neural network, i.e. information of hidden layers
        for (int l = 0; l < nn.netStruct.numLayers; l++)
            outfile << nn.hiddenLayers[l];
        return outfile;
    }

    // >> overload: load data from istream
    friend istream& operator>>(istream& infile, neuralNetworkEx& nn)
    {
        // Load epsilon, batchSize and netInfo structure
        infile >> nn.epsilon;
        infile >> nn.batchSize;
        infile >> nn.netStruct;
        // Load objective function
        string func_name;
        objFunctionMap objFuncMap;
        infile >> func_name;
        nn.objFunc = objFuncMap[func_name];
        // Load combine functions according to netInfo
        cmbFunctionMap cmbFuncMap;
        nn.cmbFuncList.resize(nn.netStruct.numLayers);
        for (int l = 0; l < nn.netStruct.numLayers; l++) {
            nn.cmbFuncList[l] = cmbFuncMap[nn.netStruct.layerInfo[l].cmbFuncName];
        }
        // Allocate memory for new settings to store information of hidden layers
        nn.hiddenLayers.resize(nn.netStruct.numLayers);
        // Load detailed settings for a neural network, i.e. information of hidden layers
        for (int l = 0; l < nn.netStruct.numLayers; l++)
            infile >> nn.hiddenLayers[l];
        return infile;
    }
    
protected:
    // Initialize the hidden layer, alloc device memory and assign initial value
    void initialize()
    {
        // Allocate device memory and initialized for output of hidden layers
        for (int l = 0; l < netStruct.numLayers; l++) {
            int inDim = hiddenLayers[l].numInputs;
            int outDim = hiddenLayers[l].numNodes;
            layerInputs.push_back(thrust::device_vector<float>(inDim * batchSize, 0));
            layerOutputs.push_back(thrust::device_vector<float>(outDim * batchSize, 0));
            layerOutputs_drv.push_back(thrust::device_vector<float>(outDim * batchSize, 0));

            // Only alloc memory when combine function is non-trivial
            cmbOutputs_drv.push_back(vector<thrust::device_vector<float>>());
            for (int p = 0; p < netStruct.parentList[l].size(); p++) {
                int parentIdx = netStruct.parentList[l][p];
                int parent_outDim = hiddenLayers[parentIdx].numNodes;
                bool non_trivial = ( !cmbFuncList[l].func.isNull() );
                cmbOutputs_drv[l].push_back(thrust::device_vector<float>( non_trivial ? parent_outDim * batchSize : 1, 0));
            }

            // Allocate memory for delta
            delta.push_back(vector<thrust::device_vector<float>>());
            // The last layer has no child, but we need compute its delta
            if (l == netStruct.numLayers - 1)
                delta[l].push_back(thrust::device_vector<float>(outDim * batchSize, 0)); 
            // The rest layers have child, just create delta branches same as child number
            for (int c = 0; c < netStruct.childList[l].size(); c++)
                delta[l].push_back(thrust::device_vector<float>(outDim * batchSize, 0)); 

            weights_drv.push_back(thrust::device_vector<float>(outDim * inDim, 0)); 
            bias_drv.push_back(thrust::device_vector<float>(outDim, 0)); 
        }
        // Allocate device memory for random input sources
        for (int s = 0; s < netStruct.numSource; s++) {
            if (netStruct.srcDist[s] != "fixed")
                temp_rand_input.push_back(thrust::device_vector<float>(netStruct.srcDim[s] * batchSize, 0));
            else
                temp_rand_input.push_back(thrust::device_vector<float>(1, 0));
        }
        // Allocate device memory for auxiliary all one vector
        one_vec = thrust::device_vector<float>(batchSize,1);
    }

    // Multiply the arrays A and B on GPU and save the result in C
    // C = alpha * op(A) * op(B) + beta * C, default is C = A*B
    void gpu_blas_mmul(const thrust::device_vector<float>& A, const thrust::device_vector<float>& B, thrust::device_vector<float>& C, 
        int final_row_dim, int final_col_dim, int shared_dim, 
        float alpha = 1, float beta = 0, 
        cublasOperation_t opA = CUBLAS_OP_N, cublasOperation_t opB = CUBLAS_OP_N) 
    {
        // Leading dimensions
        int ldA = (opA == CUBLAS_OP_N) ? final_row_dim : shared_dim;
        int ldB = (opB == CUBLAS_OP_N) ? shared_dim : final_col_dim;
        int ldC = final_row_dim;
        const float* pA = thrust::raw_pointer_cast(&A[0]);
        const float* pB = thrust::raw_pointer_cast(&B[0]);
        float* pC = thrust::raw_pointer_cast(&C[0]);
        // Do the actual multiplication
        cublasSgemm(handle, opA, opB, 
            final_row_dim, final_col_dim, shared_dim, 
            &alpha, pA, ldA, pB, ldB, 
            &beta, pC, ldC);
    }

    // Add the arrays A and B on GPU and save the result in C
    // C = alpha * op(A) + beta * op(B), default is C = A+B
    void gpu_blas_madd(const thrust::device_vector<float>& A, const thrust::device_vector<float>& B, thrust::device_vector<float>& C, 
        int final_row_dim, int final_col_dim,
        float alpha = 1, float beta = 1, 
        cublasOperation_t opA = CUBLAS_OP_N, cublasOperation_t opB = CUBLAS_OP_N) 
    {
        // Leading dimensions
        int ldA = (opA == CUBLAS_OP_N) ? final_row_dim : final_col_dim;
        int ldB = (opB == CUBLAS_OP_N) ? final_row_dim : final_col_dim;
        int ldC = final_row_dim;
        const float* pA = thrust::raw_pointer_cast(&A[0]);
        const float* pB = thrust::raw_pointer_cast(&B[0]);
        float* pC = thrust::raw_pointer_cast(&C[0]);
        // Do the actual multiplication
        cublasSgeam(handle, opA, opB, 
            final_row_dim, final_col_dim,
            &alpha, pA, ldA, &beta, pB, ldB, pC, ldC);
    }

    // Set input pointers for combine functions
    void setCmbInputPtr(vector<thrust::device_vector<float>>& inputs)
    {
        static bool flag = false;
        if (!flag) {
            // Given DAG network structure, set input pointers for combine function
            for (int layerIdx = 0; layerIdx < netStruct.numLayers; layerIdx++) 
            {
                // If current layer is connected to an input source, store its position in the combine function
                int inputPos = (netStruct.srcList[layerIdx][0] == -1) ? -1 : netStruct.srcList[layerIdx][1];
                int numParam = (int) (inputPos != -1) + (int) netStruct.parentList[layerIdx].size();
                // Construct vector of input pointers for combine function
                for (int j = 0, p = 0; j < CMB_MAX_ARG_NUM; j++) {
                    if (j == inputPos)
                        cmbFuncList[layerIdx].inputPtr[j] = &inputs[netStruct.srcList[layerIdx][0]];
                    else if (j < numParam) {
                        // Get the parent layer index of the p-th parent of current layer index
                        int parent_layerIdx = netStruct.parentList[layerIdx][p];
                        cmbFuncList[layerIdx].inputPtr[j] = &layerOutputs[parent_layerIdx];
                        p++;
                    }
                    else {
                        // Use first input pointer to fill dummy input pointers
                        cmbFuncList[layerIdx].inputPtr[j] = cmbFuncList[layerIdx].inputPtr[0];
                    }
                }
            }
        }
    }

    // Feed-forward algorithm to compute the output of network
    void feedForward(vector<thrust::device_vector<float>>& inputs, bool training = true)
    {
        // Set pointer of inputs for combine functions
        setCmbInputPtr(inputs);

        // For each stage of forward scheduler
        for (int stage = 0; stage < netStruct.schedulerFwd.size(); stage++ )
            // For each layer in the current stage
            for (int l = 0; l < netStruct.schedulerFwd[stage].size(); l++) {
                // Get the layer index
                int layerIdx = netStruct.schedulerFwd[stage][l];
                /* Compute the input of current layer */
                // If the combine function is not trivial, i.e. not identity function
                // Combine all outputs from parent layers and connected input source to compute input of the current layer
                if (!cmbFuncList[layerIdx].func.isNull()) {
                    // Set input pointer of current layer as the output pointer of combine function and then set zip iterator
                    cmbFuncList[layerIdx].setZipIterator(layerInputs[layerIdx]);
                    thrust::for_each(cmbFuncList[layerIdx].iter_begin, cmbFuncList[layerIdx].iter_end, cmbFuncList[layerIdx].func);
                }
                // If the combine function is trivial, i.e. identity function
                // Then it mean this layer is only connected to one input source, or one parent layer
                else {
                    // If current layer is only connected to an input source
                    // Then the input of current layer is this input source
                    if (netStruct.srcList[layerIdx][0] != -1)
                        thrust::copy(inputs[netStruct.srcList[layerIdx][0]].begin(), 
                            inputs[netStruct.srcList[layerIdx][0]].end(),
                            layerInputs[layerIdx].begin());
                    // If current layer is only connected to a parent layer
                    // Then the input of current layer is the output of its unique parent layer
                    else
                        thrust::copy(layerOutputs[netStruct.parentList[layerIdx][0]].begin(), 
                            layerOutputs[netStruct.parentList[layerIdx][0]].end(),
                            layerInputs[layerIdx].begin());
                }
                
                // Compute the output and its derivative of current layer
                if (training) 
                    hiddenLayers[layerIdx].computeOutputs(layerInputs[layerIdx], layerOutputs[layerIdx], layerOutputs_drv[layerIdx]);
                else
                    hiddenLayers[layerIdx].computeOutputs(layerInputs[layerIdx], layerOutputs[layerIdx]);
            }
    }

    // Back-propagation algorithm to update weights
    void backPropagation(vector<thrust::device_vector<float>>& inputs, thrust::device_vector<float>& targets)
    {
        // Initialize temporary variables
        int numLayers = netStruct.numLayers;
        vector<size_t> numNodes(numLayers);
        for (int l = 0; l < numLayers; l++)
            numNodes[l] = netStruct.layerInfo[l].numNodes;
        // cout << delta[numLayers-1][0].size() << endl;

        /* The last hidden layer depends on derivative of objective function w.r.t. the output */
        // Compute delta for the last hidden layer: delta_{L-1} = Loss^prime(beta_{L-1}) .* sigma_{L-1}^prime(alpha_{L-1})
        thrust::transform(targets.begin(), targets.end(), 
            layerOutputs[numLayers-1].begin(), 
            delta[numLayers-1][0].begin(), objFunc.drv);
        thrust::transform(delta[numLayers-1][0].begin(), delta[numLayers-1][0].end(), 
            layerOutputs_drv[numLayers-1].begin(), 
            delta[numLayers-1][0].begin(), thrust::multiplies<float>());
        // Do compute and update for W,b only if last layer is not output layer
        if (!lastLayerAsOutput) {
            // Compute dW_{L-1} = batch average of delta_{L-1} * gamma_{L-1}^T = 1/batchSize * delta_{L-1} * gamma_{L-1}^T
            gpu_blas_mmul(delta[numLayers-1][0], layerInputs[numLayers-1], weights_drv[numLayers-1], 
                    hiddenLayers[numLayers-1].numNodes, hiddenLayers[numLayers-1].numInputs, batchSize, 
                    1.f / batchSize, 0,
                    CUBLAS_OP_N, CUBLAS_OP_T);
            // Compute db_{L-1} = batch average of delta_{L-1} = 1/batchSize * delta_{L-1} * vec(1)^T
            gpu_blas_mmul(delta[numLayers-1][0], one_vec, bias_drv[numLayers-1], 
                    hiddenLayers[numLayers-1].numNodes, 1, batchSize, 
                    1.f / batchSize, 0, 
                    CUBLAS_OP_N, CUBLAS_OP_T);
            // Store derivatives to the hidden layer
            hiddenLayers[numLayers-1].setDParams(weights_drv[numLayers-1], bias_drv[numLayers-1]);
        }
        

        /* The rest hidden layers depend on derivatives of combine functions w.r.t. their child */
        // For each stage of backward scheduler
        for (int stage = 1; stage < netStruct.schedulerBk.size(); stage++)
            // For each layer in the current stage
            for (int l = 0; l < netStruct.schedulerBk[stage].size(); l++) {
                // Get the layer index
                int layerIdx = netStruct.schedulerBk[stage][l];
                /* Compute detla for current layer before multiplying sigma_l^prime(alpha_l) */
                // For each child layer of current layer, compute their delta and sum them up
                for (int c = 0; c < netStruct.childList[layerIdx].size(); c++) {
                    // Get the child layer index
                    int child_layerIdx = netStruct.childList[layerIdx][c];
                    // Compute the position of element 'layerIdx' in the vector 'parentList[child_layerIdx]'
                    int pc = netStruct.getParentListIndex(child_layerIdx,layerIdx);
                    // If there is a non-trival combine function between layers 'layerIdx' and 'child_layerIdx'
                    // The partial derivative w.r.t. output of layer 'layerIdx' need to be included
                    if (!cmbFuncList[child_layerIdx].func.isNull()) {
                        // Compute the partial derivative of the combine function of layer 'child_layerIdx' w.r.t. layer 'layerIdx'
                        // i.e. partial phi_{l_c} / partial beta_l
                        cmb_functor cmb_drv_func = cmbFuncList[child_layerIdx].drv[pc];
                        cmbFuncList[child_layerIdx].setZipIterator(cmbOutputs_drv[child_layerIdx][pc]);
                        thrust::for_each(cmbFuncList[child_layerIdx].iter_begin, cmbFuncList[child_layerIdx].iter_end, cmb_drv_func);
                        // Compute c-th part of delta_l: c-th delta_l = W_{l_c}^T * delta_{l_c} .* (partial phi_{l_c} / partial beta_l)
                        gpu_blas_mmul(hiddenLayers[child_layerIdx].weights, delta[child_layerIdx][0], delta[layerIdx][c], 
                            hiddenLayers[child_layerIdx].numInputs, batchSize, hiddenLayers[child_layerIdx].numNodes,
                            1, 0, 
                            CUBLAS_OP_T, CUBLAS_OP_N);
                        thrust::transform(delta[layerIdx][c].begin(), delta[layerIdx][c].end(), 
                            cmbOutputs_drv[child_layerIdx][pc].begin(), 
                            delta[layerIdx][c].begin(), thrust::multiplies<float>());
                    }
                    // If there is a trival combine function between layers 'layerIdx' and 'child_layerIdx'
                    // The partial derivative w.r.t. output of layer 'layerIdx' is just one and can be omitted
                    else 
                        gpu_blas_mmul(hiddenLayers[child_layerIdx].weights, delta[child_layerIdx][0], delta[layerIdx][c], 
                            hiddenLayers[layerIdx].numNodes, batchSize, hiddenLayers[child_layerIdx].numNodes,
                            1, 0, 
                            CUBLAS_OP_T, CUBLAS_OP_N);

                    // Add newly computed part into index 0 to compute the summation over child layers
                    if (c > 0)
                        gpu_blas_madd(delta[layerIdx][0], delta[layerIdx][c], delta[layerIdx][0], numNodes[layerIdx], batchSize);
                }
                /* Multiplying sigma_l^prime(alpha_l) with previous calculated delta_l */
                thrust::transform(delta[layerIdx][0].begin(), delta[layerIdx][0].end(), 
                            layerOutputs_drv[layerIdx].begin(), 
                            delta[layerIdx][0].begin(), thrust::multiplies<float>());

                /* Compute d_W, d_b for current layer */
                // Compute dW_l = batch average of delta_l * gamma_l^T = 1/batchSize * delta_l * gamma_l^T
                gpu_blas_mmul(delta[layerIdx][0], layerInputs[layerIdx], weights_drv[layerIdx], 
                    hiddenLayers[layerIdx].numNodes, hiddenLayers[layerIdx].numInputs, batchSize, 
                    1.f / batchSize, 0, 
                    CUBLAS_OP_N, CUBLAS_OP_T);
                // Compute db_l = batch average of delta_l = 1/batchSize * delta_l * vec(1)^T
                gpu_blas_mmul(delta[layerIdx][0], one_vec, bias_drv[layerIdx], 
                    hiddenLayers[layerIdx].numNodes, 1, batchSize, 
                    1.f / batchSize, 0, 
                    CUBLAS_OP_N, CUBLAS_OP_T);
                // Store derivatives to the hidden layer
                hiddenLayers[layerIdx].setDParams(weights_drv[layerIdx], bias_drv[layerIdx]);
            }
        
        // Update all parameters after computing all gradients
        for (int l = 0; l < numLayers; l++)
            hiddenLayers[l].updateParameter(epsilon);
    }
    
    // Divide data into batches, if number of data is not multiple of batch size, the remainder will be dropped
    void divide(const vector<vector<vector<float>>>& Data, const vector<vector<float>>& Target, vector<networkBatchInputEx_host>& batchedInput, bool shuffle = true)
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
            inputDims[s] = Data[s][0].size();
        size_t targetDim = Target[0].size();
        
        // Construct batch input structure vector
        for (int i = 0; i < numBatch; i++) {
            // Assign inputs values
            for (int s = 0; s < numSource; s++) {
                // For fixed input data, permute the samples and batch them
                if (!isRandom[s]) {
                    for (int b = 0; b < batchSize; b++)
                        thrust::copy(Data[s][perm[i*batchSize+b]].begin(), Data[s][perm[i*batchSize+b]].end(), batchedInput[i].input[s].begin() + b*inputDims[s]);
                }
                // For random input data, generate them according to their distribution
                else {
                    random_generator.randomGenerator(temp_rand_input[s], netStruct.srcDist[s], inputDims[s], batchSize);
                    thrust::copy(temp_rand_input[s].begin(), temp_rand_input[s].end(), batchedInput[i].input[s].begin());
                }
            }
            // Assign target values
            for (int b = 0; b < batchSize; b++)
                thrust::copy(Target[perm[i*batchSize+b]].begin(), Target[perm[i*batchSize+b]].end(), batchedInput[i].target.begin() + b*targetDim);
        }
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