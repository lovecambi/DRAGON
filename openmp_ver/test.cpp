#include <iostream>
#include <cstdlib>
#include "readfile.h"
#include "neuralNetwork.h"
#include "neuralNetworkEx.h"

bool readMNIST(string dir, vector<vector<float>>& train_image, vector<vector<float>>& test_image, vector<int>& train_label, vector<int> test_label)
{
    read_image<float>(dir+"train-images-idx3-ubyte", train_image);
    read_label(dir+"train-labels-idx1-ubyte", train_label);
    read_image<float>(dir+"t10k-images-idx3-ubyte", test_image);
    read_label(dir+"t10k-labels-idx1-ubyte", test_label);
    return !(train_image.size() == 0 || train_label.size() == 0 || test_image.size() == 0 || test_label.size() == 0);
}

int main(int argc, char ** argv){
    // Read training/testing image and labels
    string dataDir = "./";
    if (argc > 1)
        dataDir = string(argv[1]);
    vector<vector<float>> train_image, test_image;
    vector<int> train_label, test_label;
    if(!readMNIST(dataDir, train_image, test_image, train_label, test_label))
        return 0;
    cout << "Read data completed." << endl;

    
//    // Build up the neural network
//    int inputDim = 784;
//    int numLayers = 5;
//    vector<int> numNodesPerLayer = {400, 20, 20, 400, 784};
//    vector<string> trsFuncName = {"tanh", "tanh", "identity", "tanh", "sigmoid"};
//    string objFuncName = "CrossEntropy";
//    neuralNetwork<float> nn(inputDim, numLayers, numNodesPerLayer, trsFuncName, objFuncName);
//    cout << "Build networks completed." << endl;
//    
//    // Construct the input data of neural network
//    // For auto encoder, the target is same as input
//    vector<vector<float>> trainInput(train_image);
//    vector<vector<float>> trainTarget(train_image);
//    vector<vector<float>> testInput(test_image);
//    vector<vector<float>> testTarget(test_image);
    
    // Build up the neural networkEx
    vector<int> srcDim = {784, 20};
    vector<string> srcDist = {"fixed","gaussian"};
    vector<int> inputDim = {784, 400, 400, 20, 400};
    vector<int> numNodes = {400, 20, 20, 400, 784};
    vector<string> trsFuncName = {"tanh","tanh","tanh","sigmoid","sigmoid"};
    vector<string> cmbFuncName = {"null","null","null","NormRnd","null"};
    string objFuncName = "CrossEntropy";
    vector<vector<int>> I2L_edges = {{0,0},{1,3,3}};
    vector<vector<int>> L2L_edges = {{0,1},{0,2},{1,3},{2,3},{3,4}};
    neuralNetworkEx<float> nn(srcDim,srcDist,inputDim,numNodes,trsFuncName,cmbFuncName, I2L_edges, L2L_edges, objFuncName);
    cout << "Build Networks Completed." << endl;
    
    // Construct the input data of neural network
    // For auto encoder, the target is same as input
    vector<vector<vector<float>>> trainInput({train_image, vector<vector<float>>(train_image.size())});
    vector<vector<float>> trainTarget(train_image);
    vector<vector<vector<float>>> testInput({test_image, vector<vector<float>>(test_image.size())});
    vector<vector<float>> testTarget(test_image);
    
    omp_set_num_threads(8);
    // Train the neural network
    int numEpoch = 1;
    int batchSize = 1000;
    if (argc > 2)
      batchSize = atoi(argv[2]);
//    vector<vector<float>> output = nn.apply(testInput);
//    nn.train(numEpoch, batchSize, trainInput, trainTarget);
    nn.train(numEpoch, batchSize, trainInput, trainTarget, testInput, testTarget);
    
    cout << "Train Networks Completed." << endl;
    return 0;
}

