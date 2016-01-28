#ifndef AUTOENCODER_
#define AUTOENCODER_

#include "neuralNetworkEx.h"

class AutoEncoder : public neuralNetworkEx
{
    // C++11 feature: inherit cosntructor
    using neuralNetworkEx::neuralNetworkEx;
private:
    int encoderLayerIdx;                                // Layer index of input layer for decoder
    int cuttingStageIdx;                                // Cutting stage index in forward scheduler
    vector<thrust::device_vector<float>> mid_result;    // Use to store output of encoder and input of decoder
public:
    // Set the input layer for the decoder
    // The underlying neural network is splitted into lower part encoder and upper part decoder
    void setEncoderLayerIdx(int layerIdx)   
    {   
        assert(isCuttingLayer(layerIdx));
        encoderLayerIdx = layerIdx;
        size_t num_parent = netStruct.parentList[encoderLayerIdx].size();
        for (int i = 0; i < num_parent; i++) {
            int parentLayerIdx = netStruct.parentList[encoderLayerIdx][i];
            int outputDim = hiddenLayers[parentLayerIdx].numNodes;
            mid_result.push_back(thrust::device_vector<float>(outputDim * batchSize));
        }
    }

    // Given input data, use the lower half of neural network as encoder
    vector<vector<vector<float>>> encode(const vector<vector<vector<float>>>& inputData, bool print_info = true)
    {
        assert( inputData.size() > 0 );
        
        size_t numSample = inputData[0].size();
        size_t numBatch = numSample / batchSize;

        vector<size_t> inputDim(inputData.size());
        for (int s = 0; s < inputData.size(); s++)
            inputDim[s] = netStruct.srcDim[s];
        size_t targetDim = 0;

        // Initialize temporary host_vector to store encoder outputs
        size_t num_parent = netStruct.parentList[encoderLayerIdx].size();
        vector<vector<thrust::host_vector<float>>> outputs(numBatch);
        vector<size_t> enc_outputDim(num_parent);
        for (int j = 0; j < num_parent; j++) {
            int parentLayerIdx = netStruct.parentList[encoderLayerIdx][j];
            enc_outputDim[j] = hiddenLayers[parentLayerIdx].numNodes;
            for (int i = 0; i < numBatch; i++)
                outputs[i].push_back(thrust::host_vector<float>(enc_outputDim[j] * batchSize));
        }

        // Divide testing data into batches without permute
        vector<vector<float>> dummy_inputTarget(numSample);
        vector<networkBatchInputEx_host> batchesHost(numBatch, networkBatchInputEx_host(batchSize, inputDim, targetDim));
        networkBatchInputEx_device batchDevice(batchSize, inputDim, targetDim);
        divide(inputData, dummy_inputTarget, batchesHost, false);
        if (print_info)
            cout << "Divided input data into " << numBatch << " batches" << endl;
        
        // Pass the input through network to get output
        clock_t timer = clock(), time_elapse;
        for (int i = 0; i < numBatch; i++) {
            batchDevice.copy(batchesHost[i]);
            encodeBatch(batchDevice.input);

            // Copy encoder output to host_vector
            for (int j = 0; j < num_parent; j++)
                thrust::copy(mid_result[j].begin(), mid_result[j].end(), outputs[i][j].begin());
            
            time_elapse = clock() - timer;
            timer = clock();
            if (print_info)
                cout << "Batch " << i << " out of " << numBatch << " finished computation" << ", used " << (float)time_elapse / CLOCKS_PER_SEC << " sec" << endl;
        }
        
        // Convert host_vector to vector
        vector<vector<vector<float>>> encodedData(num_parent, vector<vector<float>>(numSample));
        for (int j = 0; j < num_parent; j++)
            for (int i = 0; i < numBatch; i++)
                for (int b = 0; b < batchSize; b++) {
                    encodedData[j][i*batchSize+b].resize(enc_outputDim[j]);
                    thrust::copy(outputs[i][j].begin() + b*enc_outputDim[j], outputs[i][j].begin() + (b+1)*enc_outputDim[j], encodedData[j][i*batchSize+b].begin());
                }
        if (print_info)
            cout << "Finished conversion from batches to 2D-array" << endl;
        return encodedData;
    }

    // Given encoded data, use upper half of the neural network to decode
    vector<vector<float>> decode(const vector<vector<vector<float>>>& inputData, const vector<vector<vector<float>>>& encodedData, bool print_info = true)
    {
        assert( inputData.size() > 0 );
        
        size_t numSample = inputData[0].size();
        size_t numBatch = numSample / batchSize;

        vector<size_t> inputDim(inputData.size());
        for (int s = 0; s < inputData.size(); s++)
            inputDim[s] = netStruct.srcDim[s];
        size_t targetDim = 0;
        size_t outputDim = hiddenLayers[netStruct.numLayers-1].numNodes;

        // Compute dimension to store encoder outputs
        size_t num_parent = netStruct.parentList[encoderLayerIdx].size();
        vector<size_t> enc_outputDim(num_parent);
        for (int j = 0; j < num_parent; j++) {
            int parentLayerIdx = netStruct.parentList[encoderLayerIdx][j];
            enc_outputDim[j] = hiddenLayers[parentLayerIdx].numNodes;
        }
        
        // Divide testing data into batches without permute
        vector<vector<float>> dummy_inputTarget(numSample);
        vector<networkBatchInputEx_host> batchesHost(numBatch, networkBatchInputEx_host(batchSize, inputDim, targetDim));
        vector<networkBatchInputEx_host> batchesCodeHost(numBatch, networkBatchInputEx_host(batchSize, enc_outputDim, targetDim));
        networkBatchInputEx_device batchDevice(batchSize, inputDim, targetDim);
        divide(inputData, dummy_inputTarget, batchesHost, false);
        divide(encodedData, dummy_inputTarget, batchesCodeHost, false);
        if (print_info)
            cout << "Divided input data into " << numBatch << " batches" << endl;
        
        // Pass the input through network to get output
        clock_t timer = clock(), time_elapse;
        vector<thrust::host_vector<float>> outputs(numBatch, vector<float>(outputDim*batchSize));
        for (int i = 0; i < numBatch; i++) {
            batchDevice.copy(batchesHost[i]);
            // Copy encoded data to mid_result as input of decoder
            for (int j = 0; j < num_parent; j++)
                thrust::copy(batchesCodeHost[i].input[j].begin(), batchesCodeHost[i].input[j].end(), mid_result[j].begin());

            decodeBatch(batchDevice.input);
            thrust::copy(layerOutputs[netStruct.numLayers-1].begin(), layerOutputs[netStruct.numLayers-1].end(), outputs[i].begin());
            
            time_elapse = clock() - timer;
            timer = clock();
            if (print_info)
                cout << "Batch " << i << " out of " << numBatch << " finished computation" << ", used " << (float)time_elapse / CLOCKS_PER_SEC << " sec" << endl;
        }
        
        // Convert vector<vector<batchData<T>>> back to vector<vector<T>>
        vector<vector<float>> decodedData(numSample, vector<float>(outputDim));
        for (int i = 0; i < numBatch; i++)
            for (int b = 0; b < batchSize; b++)
                thrust::copy(outputs[i].begin() + b*outputDim, outputs[i].begin() + (b+1)*outputDim, decodedData[i*batchSize+b].begin());
        if (print_info)
            cout << "Finished conversion from batches to 2D-array" << endl;
        return decodedData;
    }

    
private:
    // Only cutting layer can be set is input layer for decoder
    // A layer is a cutting layer iff in forward scheduler it is the only layer in its stage
    bool isCuttingLayer(int layerIdx)
    {
        for (int stage = 0; stage < netStruct.schedulerFwd.size(); stage++)
            if (netStruct.schedulerFwd[stage].size() == 1 && netStruct.schedulerFwd[stage][0] == layerIdx) {
                cuttingStageIdx = stage;
                return true;
            }
        return false;
    }

    // Encode the input
    void encodeBatch(vector<thrust::device_vector<float>>& inputs)
    {
        // Guarantee the output parameter has allocated memory for the right size
        assert(cuttingStageIdx > 1);
        // Set pointer of inputs for combine functions
        setCmbInputPtr(inputs);

        // For each stage of forward scheduler up to cutting stage index
        for (int stage = 0; stage < cuttingStageIdx; stage++ )
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
                
                // Compute the output and of current layer
                hiddenLayers[layerIdx].computeOutputs(layerInputs[layerIdx], layerOutputs[layerIdx]);

                // Store current layer output to the encoder output if the layer is parent of encode layer
                // Since cutting layer is the only layer in its stage, it must be the only child of its parents
                if (netStruct.childList[layerIdx][0] == encoderLayerIdx) {
                    int pc = netStruct.getParentListIndex(encoderLayerIdx,layerIdx);
                    thrust::copy(layerOutputs[layerIdx].begin(), layerOutputs[layerIdx].end(), mid_result[pc].begin());                    
                }
            }
    }

    // Decode the input
    void decodeBatch(vector<thrust::device_vector<float>>& inputs)
    {
        // Guarantee the output parameter has allocated memory for the right size
        assert(cuttingStageIdx > 1);
        // Set pointer of inputs for combine functions
        setCmbInputPtr(inputs);
        // If current layer is connected to an input source, store its position in the combine function
        int inputPos = (netStruct.srcList[encoderLayerIdx][0] == -1) ? -1 : netStruct.srcList[encoderLayerIdx][1];
        int numParam = (int) (inputPos != -1) + (int) netStruct.parentList[encoderLayerIdx].size();
        // Construct vector of input pointers for combine function
        for (int j = 0, p = 0; j < CMB_MAX_ARG_NUM; j++) {
            if (j == inputPos)
                cmbFuncList[encoderLayerIdx].inputPtr[j] = &inputs[netStruct.srcList[encoderLayerIdx][0]];
            else if (j < numParam) {
                // Get the parent layer index of the p-th parent of current layer index
                cmbFuncList[encoderLayerIdx].inputPtr[j] = &mid_result[p];
                p++;
            }
            else {
                // Use first input pointer to fill dummy input pointers
                cmbFuncList[encoderLayerIdx].inputPtr[j] = cmbFuncList[encoderLayerIdx].inputPtr[0];
            }
        }

        // For each stage of forward scheduler up to cutting stage index
        for (int stage = cuttingStageIdx; stage < netStruct.schedulerFwd.size(); stage++)
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
                    else {
                        if (layerIdx != encoderLayerIdx)
                            thrust::copy(layerOutputs[netStruct.parentList[layerIdx][0]].begin(), 
                                layerOutputs[netStruct.parentList[layerIdx][0]].end(),
                                layerInputs[layerIdx].begin());
                        else
                            thrust::copy(mid_result[0].begin(), mid_result[0].end(), layerInputs[layerIdx].begin());
                    }
                }
                
                // Compute the output and of current layer
                hiddenLayers[layerIdx].computeOutputs(layerInputs[layerIdx], layerOutputs[layerIdx]);
            }
    }
};

#endif