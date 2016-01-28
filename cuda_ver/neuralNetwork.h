#ifndef NEURALNETWORK_
#define NEURALNETWORK_

#include <cstdlib>
#include <algorithm>
#include "hiddenLayer.h"
#include "objFuncMap.h"

// Struct of network batch input for neuralNetwork class
struct networkBatchInput_host {
    thrust::host_vector<float> input;
    thrust::host_vector<float> target;
    networkBatchInput_host(size_t batchSize = 1, size_t inputDim = 1, size_t targetDim = 1) 
    {
        input = thrust::host_vector<float>(inputDim*batchSize);
        target = thrust::host_vector<float>(targetDim*batchSize);
    }
};

struct networkBatchInput_device {
    thrust::device_vector<float> input;
    thrust::device_vector<float> target;
    networkBatchInput_device(size_t batchSize = 1, size_t inputDim = 1, size_t targetDim = 1) 
    {
        input = thrust::device_vector<float>(inputDim*batchSize);
        target = thrust::device_vector<float>(targetDim*batchSize);
    }
    void copy(networkBatchInput_host& host_batch) 
    {
        input = host_batch.input;
        target = host_batch.target;
    }
};

class neuralNetwork {
private:
    int numInputs;                                              // number of input dimension
    int numLayers;                                              // number of all hidden layers in the network

    int batchSize;                                              // batch size of batch training
    cublasHandle_t handle;                                      // cuBlas handle for matrix operation
    objFuncInfo objFunc;                                        // Objective function and its partial derivative w.r.t. output
    float epsilon;                                              // Unit step size for computing adaptive step size

    vector<hiddenLayer> hiddenLayers;                           // array of hidden layers
    vector<thrust::device_vector<float>> layerOutputs;          // Use to store output of each hidden layer
    vector<thrust::device_vector<float>> layerOutputs_drv;      // Use to store output derivative of each hidden layer

    vector<thrust::device_vector<float>> delta;                 // Partial derivative of objective function w.r.t. linear outputs
    vector<thrust::device_vector<float>> weights_drv;           // Partial derivative of objective function w.r.t. weights
    vector<thrust::device_vector<float>> bias_drv;              // Partial derivative of objective function w.r.t. bias
    thrust::device_vector<float> one_vec;                       // auxiliary vector of all ones

public:
// Constructor using transfer function name
    neuralNetwork(int _numInputs, int _numLayers, vector<int>& numNodesPerLayers, int _batchSize, 
        vector<string>& trsFuncName, string objFuncName, float _epsilon = 0.01) \
    : numInputs(_numInputs), numLayers(_numLayers), batchSize(_batchSize), epsilon(_epsilon)
    {
        assert(numLayers > 0);
        trsFunctionMap trsFuncMap;
        // i-th hidden layer has numNodesPerLayers[i-1] input value and numNodesPerLayers[i] outputs
        hiddenLayers.resize(numLayers);
        for (int l = 0; l < numLayers; l++)
            hiddenLayers[l] = hiddenLayer((l==0) ? numInputs : numNodesPerLayers[l - 1], numNodesPerLayers[l], batchSize, trsFuncName[l]);
        objFunctionMap objFuncMap;
        objFunc = objFuncMap[objFuncName];
        // Create a handle for CUBLAS
        cublasCreate(&handle);
        // Malloc memory for layer input and output
        initialize();
    }
    
    // Constructor from file
    neuralNetwork(string filename)
    {
        this->loadFile(filename);
        // Create a handle for CUBLAS
        cublasCreate(&handle);
        // Malloc memory for layer input and output
        initialize();
    }

	// Destructor
	virtual ~neuralNetwork()
    {
        // Destroy the handle
        cublasDestroy(handle);
        // device_vector are automatically released by cudaFree()
	}

	// Assign weights for all edges in the network
	void setParams(vector<thrust::device_vector<float>>& weights, vector<thrust::device_vector<float>> bias)
    {
		for (int i = 0; i < numLayers; i++)
			hiddenLayers[i].setParams(weights[i], bias[i]);
	}
    
    // Train the neural network and store training/testing MSE for each epoch
    vector<vector<float>> train(int epoch, const vector<vector<float>>& trainData, const vector<vector<float>>& trainTarget, const vector<vector<float>>& testData, const vector<vector<float>>& testTarget, bool print_info = true)
    {
        // Guarantee that the sample number for input and target are same
        assert( trainData.size() == trainTarget.size() );
        assert( testData.size() == testTarget.size() );
        
        size_t numTrainSample = trainTarget.size();
        size_t numTrainBatch = numTrainSample / batchSize;
        size_t numTestSample = testTarget.size();
        size_t numTestBatch = numTestSample / batchSize;

        size_t inputDim = trainData[0].size();
        size_t targetDim = trainTarget[0].size();

        // Initialize vector to store loss for each epoch
        vector<float> trainLoss(epoch,0);
        vector<float> testLoss(epoch,0);
        
        // Divide testing data into batches
        if (print_info)
            cout << "Permute testing samples and divide them into batches" << endl;
        vector<networkBatchInput_host> trainBatchesHost(numTrainBatch, networkBatchInput_host(batchSize, inputDim, targetDim));
        vector<networkBatchInput_host> testBatchesHost(numTestBatch, networkBatchInput_host(batchSize, inputDim, targetDim));
        networkBatchInput_device trainBatchDevice(batchSize, inputDim, targetDim);
        networkBatchInput_device testBatchDevice(batchSize, inputDim, targetDim);
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
                outputs = layerOutputs[numLayers-1];
                temp = thrust::inner_product(trainBatchDevice.target.begin(), trainBatchDevice.target.end(), 
                    outputs.begin(), 0.0f, thrust::plus<float>(), objFunc.func);
                trainLoss[t] += temp;
                
                time_elapse = clock() - timer;
                timer = clock();
                if (print_info)
                    cout << "Epoch " << t << ", batch " << i << ", train current batch loss = " << temp / batchSize << ", used " << (float)time_elapse / CLOCKS_PER_SEC << " sec" << endl;
            }
            // Test the network using testing data and keep track of testing error
            for (int i = 0; i < numTestBatch; i++) {
                testBatchDevice.copy(testBatchesHost[i]);
                feedForward(testBatchDevice.input);
                outputs = layerOutputs[numLayers-1];
                temp = thrust::inner_product(testBatchDevice.target.begin(), testBatchDevice.target.end(), 
                    outputs.begin(), 0.0f, thrust::plus<float>(), objFunc.func);
                testLoss[t] += temp;
                
                time_elapse = clock() - timer;
                timer = clock();
                if (print_info)
                    cout << "Epoch " << t << ", batch " << i << ", test current batch loss = " << temp / batchSize << ", used " << (float)time_elapse / CLOCKS_PER_SEC << " sec" << endl;
            }
            cout << "Epoch " << t << ", train loss = " << trainLoss[t] / trainData.size() << ",\t test loss = " << testLoss[t] / testData.size() << endl;
        }
        return {trainLoss, testLoss};
    }
    
    // Train the neural network and store training MSE, do not use testing data during training procedure
    vector<float> train(int epoch, const vector<vector<float>>& trainData, const vector<vector<float>>& trainTarget, bool print_info = true)
    {
        // Guarantee that the sample number for input and target are same
        assert( trainData.size() == trainTarget.size() );
        
        size_t numTrainSample = trainTarget.size();
        size_t numTrainBatch = numTrainSample / batchSize;
        
        size_t inputDim = trainData[0].size();
        size_t targetDim = trainTarget[0].size();

        // Initialize vector to store loss for each epoch
        vector<float> trainLoss(epoch,0);
        
        vector<networkBatchInput_host> trainBatchesHost(numTrainBatch, networkBatchInput_host(batchSize, inputDim, targetDim));
        networkBatchInput_device trainBatchDevice(batchSize, inputDim, targetDim);
        
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
                outputs = layerOutputs[numLayers-1];
                temp = thrust::inner_product(trainBatchDevice.target.begin(), trainBatchDevice.target.end(), 
                    outputs.begin(), 0.0f, thrust::plus<float>(), objFunc.func);
                trainLoss[t] += temp;
                
                time_elapse = clock() - timer;
                timer = clock();
                if (print_info)
                    cout << "Epoch " << t << ", batch " << i << ", train current batch loss = " << temp / batchSize << ", used " << (float)time_elapse / CLOCKS_PER_SEC << " sec" << endl;
            }
            cout << "Epoch " << t << ", running train loss = " << trainLoss[t] / numTrainSample << endl;
        }
        return trainLoss;
    }

    
    // Given input data, use the neural network to generate the output
    vector<vector<float>> apply(const vector<vector<float>>& inputData, bool print_info = true)
    {
        assert( inputData.size() > 0 );
        
        size_t numSample = inputData.size();
        size_t numBatch = numSample / batchSize;
        
        size_t inputDim = inputData[0].size();
        size_t targetDim = 0;
        size_t outputDim = hiddenLayers[numLayers-1].numNodes;

        // Divide testing data into batches without permute
        vector<vector<float>> dummy_inputTarget(numSample);
        vector<networkBatchInput_host> batchesHost(numBatch, networkBatchInput_host(batchSize, inputDim, targetDim));
        networkBatchInput_device batchDevice(batchSize, inputDim, targetDim);
        divide(inputData, dummy_inputTarget, batchesHost, false);
        if (print_info)
            cout << "Divided input data into " << numBatch << " batches" << endl;
        
        // Pass the input through network to get output
        clock_t timer, time_elapse;
        vector<thrust::host_vector<float>> outputs(numBatch, vector<float>(outputDim*batchSize));
        for (int i = 0; i < numBatch; i++) {
            batchDevice.copy(batchesHost[i]);
            feedForward(batchDevice.input);
            thrust::copy(layerOutputs[numLayers-1].begin(), layerOutputs[numLayers-1].end(), outputs[i].begin());
            
            time_elapse = clock() - timer;
            timer = clock();
            if (print_info)
                cout << "Batch " << i << " out of " << numBatch << " finished computation" << ", used " << (float)time_elapse / CLOCKS_PER_SEC << " sec" << endl;
        }
        
        // Since the outputs store data as a column, and cuBlas stores matrix in column major, just need to split the output
        vector<vector<float>> outputData(numSample, vector<float>(outputDim));
        for (int i = 0; i < numBatch; i++)
            for (int b = 0; b < batchSize; b++)
                thrust::copy(outputs[i].begin() + b*outputDim, outputs[i].begin() + (b+1)*outputDim, outputData[i*batchSize+b].begin());
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
        outfile << numInputs << " " << numLayers << " " << batchSize << " " << epsilon << endl;
        
        // Save objective function name
        objFunctionMap objFuncMap;
        outfile << objFuncMap(objFunc.func) << endl;
        
        // Save detailed settings for a neural network, i.e. information of hidden layers
        for (int l = 0; l < numLayers; l++)
            outfile << hiddenLayers[l];
        
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
        infile >> numInputs >> numLayers >> batchSize >> epsilon;
        
        // Load objective function
        string func_name;
        objFunctionMap objFuncMap;
        infile >> func_name;
        objFunc = objFuncMap[func_name];
        
        // Allocate memory for new settings to store information of hidden layers
        hiddenLayers.resize(numLayers);
        // Load detailed settings for a neural network, i.e. information of hidden layers
        for (int l = 0; l < numLayers; l++)
            infile >> hiddenLayers[l];

        infile.close();
        return 0;
    }

private:
    // Initialize the hidden layer, alloc device memory and assign initial value
    void initialize()
    {
        // Allocate device memory and initialized for output of hidden layers
        for (int l = 0; l < numLayers; l++)
        {
            int inDim = hiddenLayers[l].numInputs;
            int outDim = hiddenLayers[l].numNodes;
            layerOutputs.push_back(thrust::device_vector<float>(outDim * batchSize, 0));
            layerOutputs_drv.push_back(thrust::device_vector<float>(outDim * batchSize, 0));
            delta.push_back(thrust::device_vector<float>(outDim * batchSize, 0)); 
            weights_drv.push_back(thrust::device_vector<float>(outDim * inDim, 0)); 
            bias_drv.push_back(thrust::device_vector<float>(outDim, 0)); 
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

	// Feed-forward algorithm to compute the output of network
	void feedForward(thrust::device_vector<float>& inputs, bool training = true)
    {
        if (training)
        {
            hiddenLayers[0].computeOutputs(inputs, layerOutputs[0], layerOutputs_drv[0]);
            for (int l = 1; l < numLayers; l++)
                hiddenLayers[l].computeOutputs(layerOutputs[l-1],layerOutputs[l], layerOutputs_drv[l]);
        }
        else
        {
            hiddenLayers[0].computeOutputs(inputs, layerOutputs[0]);
            for (int l = 1; l < numLayers; l++)
                hiddenLayers[l].computeOutputs(layerOutputs[l-1], layerOutputs[l]);
        }
    }

	// Back-propagation algorithm to update weights
	void backPropagation(thrust::device_vector<float>& inputs, thrust::device_vector<float>& targets)
    {
        // Compute delta for the last hidden layer: delta_{L-1} = Loss^prime(beta_{L-1}) .* sigma_{L-1}^prime(alpha_{L-1})
        thrust::transform(targets.begin(), targets.end(), 
            layerOutputs[numLayers-1].begin(), 
            delta[numLayers-1].begin(), objFunc.drv);
        thrust::transform(delta[numLayers-1].begin(), delta[numLayers-1].end(), 
            layerOutputs_drv[numLayers-1].begin(), 
            delta[numLayers-1].begin(), thrust::multiplies<float>());
        // Compute dW_{L-1} = batch average of delta_{L-1} * gamma_{L-1}^T = 1/batchSize * delta_{L-1} * gamma_{L-1}^T
        gpu_blas_mmul(delta[numLayers-1], layerOutputs[numLayers-2], weights_drv[numLayers-1], 
                hiddenLayers[numLayers-1].numNodes, hiddenLayers[numLayers-1].numInputs, batchSize, 
                1.f / batchSize, 0,
                CUBLAS_OP_N, CUBLAS_OP_T);
        // Compute db_{L-1} = batch average of delta_{L-1} = 1/batchSize * delta_{L-1} * vec(1)^T
        gpu_blas_mmul(delta[numLayers-1], one_vec, bias_drv[numLayers-1], 
                hiddenLayers[numLayers-1].numNodes, 1, batchSize, 
                1.f / batchSize, 0, 
                CUBLAS_OP_N, CUBLAS_OP_T);
        // Store derivatives to the hidden layer
        hiddenLayers[numLayers-1].setDParams(weights_drv[numLayers-1], bias_drv[numLayers-1]);

        for (int l = numLayers-2; l >= 0; l--) {
            // Compute delta for the l-th hidden layer: delta_l = W_{l+1}^T * delta_{l+1} .* sigma_l^prime(alpha_l)
            gpu_blas_mmul(hiddenLayers[l+1].weights, delta[l+1], delta[l], 
                hiddenLayers[l].numNodes, batchSize, hiddenLayers[l+1].numNodes,
                1, 0, 
                CUBLAS_OP_T, CUBLAS_OP_N);
            thrust::transform(delta[l].begin(), delta[l].end(), 
                layerOutputs_drv[l].begin(), 
                delta[l].begin(), thrust::multiplies<float>());
            // Compute dW_l = batch average of delta_l * gamma_l^T = 1/batchSize * delta_l * gamma_l^T
            gpu_blas_mmul(delta[l], (l==0) ? inputs : layerOutputs[l-1], weights_drv[l], 
                hiddenLayers[l].numNodes, hiddenLayers[l].numInputs, batchSize, 
                1.f / batchSize, 0, 
                CUBLAS_OP_N, CUBLAS_OP_T);
            // Compute db_l = batch average of delta_l = 1/batchSize * delta_l * vec(1)^T
            gpu_blas_mmul(delta[l], one_vec, bias_drv[l], 
                hiddenLayers[l].numNodes, 1, batchSize, 
                1.f / batchSize, 0, 
                CUBLAS_OP_N, CUBLAS_OP_T);
            // Store derivatives to the hidden layer
            hiddenLayers[l].setDParams(weights_drv[l], bias_drv[l]);
        } 
		
        // Update all parameters after computing all gradients
		for (int l = 0; l < numLayers; l++)
            hiddenLayers[l].updateParameter(epsilon);
	}
    
    // Divide data into batches, if number of data is not multiple of batch size, the remainder will be dropped
    void divide(const vector<vector<float>>& Data, const vector<vector<float>>& Target, vector<networkBatchInput_host>& batchedInput, bool shuffle = true)
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
        for (int i = 0; i < numBatch; i++) {
            // Copy permuted data and target as column (cuBlas store matrix in column major)
            for (int b = 0; b < batchSize; b++) {
                thrust::copy(Data[perm[i*batchSize+b]].begin(), Data[perm[i*batchSize+b]].end(), batchedInput[i].input.begin() + b*inputDim);
                thrust::copy(Target[perm[i*batchSize+b]].begin(), Target[perm[i*batchSize+b]].end(), batchedInput[i].target.begin() + b*targetDim);
            }
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