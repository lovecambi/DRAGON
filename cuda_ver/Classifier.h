#ifndef CLASSIFIER_
#define CLASSIFIER_

#include "neuralNetworkEx.h"

// For two tuple (value1, label1), (value2, label2), return the tuple with larger output value
typedef thrust::tuple<float,int> ValueLabel;
struct cmp_by_val
{ 
    cmp_by_val() {}
    __host__ __device__ ValueLabel operator()(ValueLabel t1, ValueLabel t2) const {
        float val1 = thrust::get<0>(t1);
        float val2 = thrust::get<0>(t2);
        return (val1 > val2) ? t1 : t2;
    }
};
// For two tuple (value1, label1), (value2, label2), return 1 if their labels are different
struct count_diff
{ 
    count_diff() {}
    __host__ __device__ int operator()(ValueLabel t1, ValueLabel t2) const {
        return (int)( thrust::get<1>(t1) != thrust::get<1>(t2) );
    }
};

class Classifier : public neuralNetworkEx
{
    // C++11 feature: inherit cosntructor
    using neuralNetworkEx::neuralNetworkEx;
private:
    // Construct auxiliary vectors for find index of maximum output
    thrust::device_vector<int> sample_index;            // Index vector: [0,0,...0,  1,1,...1,  ......,B-1,B-1,...,B-1]
    thrust::device_vector<int> label_value;             // Label vector: [0,1,...C-1,0,1,...C-1,......,0,  1,  ...,C-1]
    thrust::device_vector<int> dummy_vec;               // Dummy, only used in thrust::reduce_by_key as an output
    thrust::device_vector<ValueLabel> output_label;     // Vector of (value, label) tuple of network output
    thrust::device_vector<ValueLabel> target_label;     // Vector of (value, label) tuple of target, the values are 0 or 1
public:
    // Train the neural network and store training/testing MSE for each epoch
    vector<vector<float>> train(int epoch, const vector<vector<vector<float>>>& trainData, const vector<int>& trainLabel, const vector<vector<vector<float>>>& testData, const vector<int>& testLabel, bool print_info = true)
    {
        // Initialize auxiliary variables
        classifierInit();

        // From label to construct target
        int num_class = hiddenLayers[netStruct.numLayers-1].numNodes;
        vector<vector<float>> trainTarget(trainLabel.size(), vector<float>(num_class,0));
        for (int i = 0; i < trainLabel.size(); i++)
            trainTarget[i][trainLabel[i]] = 1;
        vector<vector<float>> testTarget(testLabel.size(), vector<float>(num_class,0));
        for (int i = 0; i < testLabel.size(); i++)
            testTarget[i][testLabel[i]] = 1;

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
        vector<float> trainErrRate(epoch,0);
        vector<float> testErrRate(epoch,0);
        
        // Divide testing data into batches: batch number index, input index, batchData index
        if (print_info)
            cout << "Permute testing samples and divide them into batches" << endl;
        vector<networkBatchInputEx_host> trainBatchesHost(numTrainBatch, networkBatchInputEx_host(batchSize, inputDim, targetDim));
        vector<networkBatchInputEx_host> testBatchesHost(numTestBatch, networkBatchInputEx_host(batchSize, inputDim, targetDim));
        networkBatchInputEx_device trainBatchDevice(batchSize, inputDim, targetDim);
        networkBatchInputEx_device testBatchDevice(batchSize, inputDim, targetDim);
        divide(testData, testTarget, testBatchesHost, false);

        thrust::device_vector<float> outputs;
        int temp_num_err = 0;
        float temp_loss = 0;
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
                temp_loss = thrust::inner_product(trainBatchDevice.target.begin(), trainBatchDevice.target.end(), 
                    outputs.begin(), 0.0f, thrust::plus<float>(), objFunc.func);
                trainLoss[t] += temp_loss / numTrainSample;
               
                // Compute number of errors in this batch
                temp_num_err = computeBatchError(outputs, trainBatchDevice.target);
                trainErrRate[t] += (float) temp_num_err / numTrainSample;

                time_elapse = clock() - timer;
                timer = clock();
                if (print_info)
                    cout << "Epoch " << t << ", batch " << i << ", average train loss = " << temp_loss / batchSize << ", " << temp_num_err \
                        << " classification error, used " << (float)time_elapse / CLOCKS_PER_SEC << " sec" << endl;
            }
            // Test the network using testing data and keep track of testing error
            for (int i = 0; i < numTestBatch; i++) {
                testBatchDevice.copy(testBatchesHost[i]);
                feedForward(testBatchDevice.input);
                outputs = layerOutputs[netStruct.numLayers-1];
                temp_loss = thrust::inner_product(testBatchDevice.target.begin(), testBatchDevice.target.end(), 
                    outputs.begin(), 0.0f, thrust::plus<float>(), objFunc.func);
                testLoss[t] += temp_loss / numTestSample;

                // Compute number of errors in this batch
                temp_num_err = computeBatchError(outputs, testBatchDevice.target);
                testErrRate[t] += (float) temp_num_err / numTestSample;
                
                time_elapse = clock() - timer;
                timer = clock();
                if (print_info)
                    cout << "Epoch " << t << ", batch " << i << ", average test loss = " << temp_loss / batchSize << ", " << temp_num_err \
                        << " classification error, used " << (float)time_elapse / CLOCKS_PER_SEC << " sec" << endl;
            }
            cout << "Epoch " << t << ",\ttrain loss = " << trainLoss[t] << ",\ttest loss = " << testLoss[t] \
                << ",\ttrain error rate = " << trainErrRate[t] << ",\ttest error rate = " << testErrRate[t] << endl;
        }
        return {trainLoss, testLoss, trainErrRate, testErrRate};
    }

    // Train the neural network and store training MSE, do not use testing data during training procedure
    vector<float> train(int epoch, const vector<vector<vector<float>>>& trainData, const vector<int>& trainLabel, bool print_info = true)
    {
        // Initialize auxiliary variables
        classifierInit();

        // From label to construct target
        int num_class = hiddenLayers[netStruct.numLayers-1].numNodes;
        vector<vector<float>> trainTarget(trainLabel.size(), vector<float>(num_class,0));
        for (int i = 0; i < trainLabel.size(); i++)
            trainTarget[i][trainLabel[i]] = 1;

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
        vector<float> trainErrRate(epoch,0);

        vector<networkBatchInputEx_host> trainBatchesHost(numTrainBatch, networkBatchInputEx_host(batchSize, inputDim, targetDim));
        networkBatchInputEx_device trainBatchDevice(batchSize, inputDim, targetDim);
        
        thrust::device_vector<float> outputs;
        int temp_num_err = 0;
        float temp_loss = 0;
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
                temp_loss = thrust::inner_product(trainBatchDevice.target.begin(), trainBatchDevice.target.end(), 
                    outputs.begin(), 0.0f, thrust::plus<float>(), objFunc.func);
                trainLoss[t] += temp_loss / numTrainSample;
                
                // Compute number of errors in this batch
                temp_num_err = computeBatchError(outputs, trainBatchDevice.target);
                trainErrRate[t] += (float) temp_num_err / numTrainSample;

                time_elapse = clock() - timer;
                timer = clock();
                if (print_info)
                    cout << "Epoch " << t << ", batch " << i << ", average test loss = " << temp_loss / batchSize << ", " << temp_num_err \
                        << " classification error, used " << (float)time_elapse / CLOCKS_PER_SEC << " sec" << endl;
            }
            cout  << "Epoch " << t << ",\trunning train loss = " << trainLoss[t] << endl;
        }
        return trainLoss;
    }
    
    // Given input data, use the neural network to predict its class
    vector<int> apply(const vector<vector<vector<float>>>& inputData, bool print_info = true)
    {
        // Initialize auxiliary variables
        classifierInit();

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
        if (print_info) {
            cout << "Divided input data into " << numBatch << " batches" << endl;            
        }
        
        // Pass the input through network to get output
        clock_t timer = clock(), time_elapse;
        vector<int> outputLabel(numSample);
        for (int i = 0; i < numBatch; i++) {
            batchDevice.copy(batchesHost[i]);
            feedForward(batchDevice.input);
            // Convert output to label
            thrust::reduce_by_key(sample_index.begin(), sample_index.end(), 
                thrust::make_zip_iterator(thrust::make_tuple(layerOutputs[netStruct.numLayers-1].begin(), label_value.begin())),
                dummy_vec.begin(), output_label.begin(), thrust::equal_to<int>(), cmp_by_val());
            // Store the labels in the output vector
            for (int b = 0; b < batchSize; b++) {
                ValueLabel temp = output_label[b];
                outputLabel[i*batchSize+b] = (int) thrust::get<1>(temp);
            }
            
            time_elapse = clock() - timer;
            timer = clock();
            if (print_info) {
                cout << "Batch " << i << " out of " << numBatch << " finished computation" << ", used " << (float)time_elapse / CLOCKS_PER_SEC << " sec" << endl;                
            }
        }
        return outputLabel;
    }

private:
    void classifierInit()
    {
        int num_class = hiddenLayers[netStruct.numLayers-1].numNodes;
        sample_index.resize(num_class*batchSize);
        label_value.resize(num_class*batchSize);
        for (int i = 0; i < num_class*batchSize; i++) {
            sample_index[i] = i / num_class;
            label_value[i] = i % num_class;
        }
        dummy_vec.resize(batchSize);
        output_label.resize(batchSize);
        target_label.resize(batchSize);
    }

    int computeBatchError(thrust::device_vector<float>& outputs, thrust::device_vector<float>& targets)
    {
        // Compute predicted labels
        thrust::reduce_by_key(sample_index.begin(), sample_index.end(), 
            thrust::make_zip_iterator(thrust::make_tuple(outputs.begin(), label_value.begin())),
            dummy_vec.begin(), output_label.begin(), thrust::equal_to<int>(), cmp_by_val());
        // Compute true labels
        thrust::reduce_by_key(sample_index.begin(), sample_index.end(), 
            thrust::make_zip_iterator(thrust::make_tuple(targets.begin(), label_value.begin())),
            dummy_vec.begin(), target_label.begin(), thrust::equal_to<int>(), cmp_by_val());
        // Compute number of errors in this batch
        int err_num = thrust::inner_product(output_label.begin(), output_label.end(), 
            target_label.begin(), 0, thrust::plus<float>(), count_diff());
        return err_num;
    }
};

#endif