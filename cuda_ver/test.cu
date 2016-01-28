#include <iostream>
#include <cstdlib>
#include <ctime>
#include "readfile.h"
#include "neuralNetwork.h"
#include "AutoEncoder.h"
#include "Classifier.h"

bool readMNIST(string dir, vector<vector<float>>& train_image, vector<vector<float>>& test_image, vector<int>& train_label, vector<int>& test_label)
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


	string type = "nnEx";
	if (argc > 2)
		type = string(argv[2]);

	if (type == "nn") {
		cout << "Using neuralNetwork class now" << endl;
		// Build up the neural network
		int inputDim = 784;
		int numLayers = 5;
		int batchSize = 100;
		vector<int> numNodesPerLayer({400, 20, 20, 400, 784});
		vector<string> trsFuncName({"tanh", "tanh", "identity", "tanh", "sigmoid"});
		string objFuncName = "CrossEntropy";
		neuralNetwork nn(inputDim, numLayers, numNodesPerLayer, batchSize, trsFuncName, objFuncName);
		cout << "Build networks completed." << endl;

		// Construct the input data of neural network
		// For auto encoder, the target is same as input
		vector<vector<float>> trainInput(train_image);
		vector<vector<float>> trainTarget(train_image);
		vector<vector<float>> testInput(test_image);
		vector<vector<float>> testTarget(test_image);	
		cout << "Construct input and target completed." << endl;

		// Train the neural network
		int numEpoch = 100;
		bool print_info = false;
		clock_t time = clock();
		// vector<vector<float>> output = nn.apply(testInput);
		// nn.train(numEpoch, trainInput, trainTarget, print_info);
	    nn.train(numEpoch, trainInput, trainTarget, testInput, testTarget, print_info);
	    cout << "Training " << numEpoch << " epoches takes " << (float)(clock()-time) / CLOCKS_PER_SEC << " seconds." << endl;

		cout << "Train Networks Completed." << endl;
	}
	else if (type == "nnEx") {
		cout << "Using neuralNetworkEx class now" << endl;
		// Build up the neural networkEx
		vector<int> srcDim({784, 20, 784});
		vector<string> srcDist({"fixed","gaussian","gaussian"});
		vector<int> inputDim({784, 400, 400, 20, 40, 40, 784});
		vector<int> numNodes({400, 20, 20, 40, 784, 784, 784});
		int batchSize = 100;
		vector<string> trsFuncName({"tanh","tanh","identity","tanh","tanh","identity","sigmoid"});
		vector<string> cmbFuncName({"null","null","null","NormRnd","null","null","NormRnd"});
		string objFuncName = "CrossEntropy";
		vector<vector<int>> I2L_edges({{0,0},{1,3,3},{2,6,3}});
		vector<vector<int>> L2L_edges({{0,1},{0,2},{1,3},{2,3},{3,4},{3,5},{4,6},{5,6}});
		float epsilon = 0.01;
		bool lastLayerAsOutput = true;
		neuralNetworkEx nn(srcDim, srcDist, inputDim, numNodes, batchSize, trsFuncName, cmbFuncName,
		 I2L_edges, L2L_edges, objFuncName, epsilon, lastLayerAsOutput);
		cout << "Build Networks Completed." << endl;

		// Construct the input data of neural network
		// For auto encoder, the target is same as input
		vector<vector<vector<float>>> trainInput({train_image, vector<vector<float>>(train_image.size()), vector<vector<float>>(train_image.size())});
		vector<vector<float>> trainTarget(train_image);
		vector<vector<vector<float>>> testInput({test_image, vector<vector<float>>(test_image.size()), vector<vector<float>>(test_image.size())});
		vector<vector<float>> testTarget(test_image);
		cout << "Construct input and target completed." << endl;

		// Train the neural network
		int numEpoch = 100;
		bool print_info = true;
		clock_t time = clock();
		// vector<vector<float>> output = nn.apply(testInput, print_info);
		nn.train(numEpoch, trainInput, trainTarget, print_info);
		// nn.train(numEpoch, trainInput, trainTarget, testInput, testTarget, print_info);
		cout << "Training " << numEpoch << " epoches takes " << (float)(clock()-time) / CLOCKS_PER_SEC << " seconds." << endl;

		cout << "Train Networks Completed." << endl;
		
	}
	else if (type == "ae") {
		// Works to be done, AutoEncoder class inherited from neuralNetworkEx class
		cout << "Using AutoEncoder class now" << endl;
		// Build up the neural networkEx
		int code_len = 20;
		vector<int> srcDim({784, code_len});
		vector<string> srcDist({"fixed","gaussian"});
		vector<int> inputDim({784, 400, 400, code_len, 400});
		vector<int> numNodes({400, code_len, code_len, 400, 784});
		int batchSize = 100;
		vector<string> trsFuncName({"tanh","tanh","tanh","sigmoid","sigmoid"});
		vector<string> cmbFuncName({"null","null","null","NormRnd","null"});
		string objFuncName = "CrossEntropy";
		vector<vector<int>> I2L_edges({{0,0},{1,3,3}});
		vector<vector<int>> L2L_edges({{0,1},{0,2},{1,3},{2,3},{3,4}});
		float epsilon = 0.01;
		bool lastLayerAsOutput = false;
		AutoEncoder nn_a(srcDim,srcDist,inputDim,numNodes,batchSize,trsFuncName,cmbFuncName, I2L_edges, L2L_edges, objFuncName, epsilon, lastLayerAsOutput);
		
		// AutoEncoder nn_a(srcDim,srcDist,inputDim,numNodes,batchSize,trsFuncName,cmbFuncName, I2L_edges, L2L_edges, objFuncName, epsilon, lastLayerAsOutput);
		cout << "Build Networks Completed." << endl;

		// Construct the input data of neural network
		// For auto encoder, the target is same as input
		vector<vector<vector<float>>> trainInput({train_image, vector<vector<float>>(train_image.size())});
		vector<vector<float>> trainTarget(train_image);
		vector<vector<vector<float>>> testInput({test_image, vector<vector<float>>(test_image.size())});
		vector<vector<float>> testTarget(test_image);
		cout << "Construct input and target completed." << endl;

		// Train the neural network
		int numEpoch = 100;
		bool print_info = false;
		clock_t time = clock();
		nn_a.setEncoderLayerIdx(3);
		cout << "Set encoder layer completed." << endl;
		nn_a.train(numEpoch, trainInput, trainTarget, print_info);
		cout << "train completed." << endl;
		vector<vector<float>> output = nn_a.apply(testInput, print_info);
		cout << "apply completed." << endl;
		vector<vector<vector<float>>> encoder_output = nn_a.encode(testInput, print_info);
		cout << "encode completed. " << endl;
		vector<vector<float>> decoder_output = nn_a.decode(testInput, encoder_output, print_info);
		cout << "decode completed." << endl;
		// nn_a.train(numEpoch, trainInput, trainTarget, testInput, testTarget, print_info);
		cout << "Training " << numEpoch << " epoches takes " << (float)(clock()-time) / CLOCKS_PER_SEC << " seconds." << endl;

		vector<int> index = {3,2,1,18,4,8,11,0,84,7};
		for (int k = 0; k < 10; k++) {
			cout << "label is " << test_label[index[k]] << ", encoded as: " << endl; 
			for (int i = 0; i < code_len; i++)
				cout << encoder_output[0][index[k]][i] << " ";
			cout << endl;
			for (int i = 0; i < 28; i++) {
				for (int j = 0; j < 28; j++)
					if (decoder_output[k][i*28+j] > 0.2)
						cout << "# ";
					else
						cout << "  ";
				cout << endl;
			}
		}

		// vector<vector<vector<float>>> custom_input(testInput);
		// custom_input[0].resize(400);
		// custom_input[1].resize(400);
		// vector<vector<vector<float>>> custom_code(2, vector<vector<float>>(400, vector<float>(2)));
		// for (int i = 0; i < 20; i++)
		// 	for (int j = 0; j < 20; j++) {
		// 		custom_code[0][i*10+j][0] = (float)(i-10) / 10;
		// 		custom_code[0][i*10+j][1] = (float)(j-10) / 10;
		// 		custom_code[1][i*10+j][0] = -100;
		// 		custom_code[1][i*10+j][1] = -100;
		// 	}
			
		// decoder_output = nn_a.decode(custom_input, custom_code, print_info);
		// cout << "Decode custom code finished" << endl;

		// for (float s1 = 0 , s2 = 0; s1 > -1.01 && s2 > -1.01 && s1 < 0.91 && s2 < 0.91; ) 
		// {
		// 	cout << "Please input s1 and s2 in range [-1, 0.9]:" << endl;
		// 	cin >> s1 >> s2;
		// 	int idx = (s1*10+10) * 20 + (s2*10+10);
		// 	cout << "s1 = " << s1 << ", s2 = " << s2 << ", take " << idx << "-th picture" << endl;
		// 	for (int i = 0; i < 28; i++) {
		// 		for (int j = 0; j < 28; j++)
		// 			if (decoder_output[idx][i*28+j] > 0.5)
		// 				cout << "# ";
		// 			else
		// 				cout << "  ";
		// 		cout << endl;
		// 	}
		// }

		cout << "Train Networks Completed." << endl;
	}
	else if (type == "cls") {
		cout << "Using Classifier class now" << endl;
		// Build up the neural networkEx
		vector<int> srcDim({784});
		vector<string> srcDist({"fixed"});
		vector<int> inputDim({784, 300});
		vector<int> numNodes({300, 10});
		int batchSize = 100;
		vector<string> trsFuncName({"tanh","sigmoid"});
		vector<string> cmbFuncName({"null","null"});
		string objFuncName = "SquareError";
		vector<vector<int>> I2L_edges({{0,0}});
		vector<vector<int>> L2L_edges({{0,1}});
		Classifier nn_c(srcDim,srcDist,inputDim,numNodes,batchSize,trsFuncName,cmbFuncName, I2L_edges, L2L_edges, objFuncName);
		cout << "Build Classifier Completed." << endl;

		// Construct the input data of neural network
		vector<vector<vector<float>>> trainInput({train_image});
		vector<int> trainLabel(train_label);
		vector<vector<vector<float>>> testInput({test_image});
		vector<int> testLabel(test_label);
		cout << "Construct input and target completed." << endl;

		// Train the neural network
		int numEpoch = 10;
		bool print_info = false;
		clock_t time = clock();
		vector<int> outputLabel1 = nn_c.apply(testInput, print_info);
		nn_c.train(numEpoch, trainInput, trainLabel, print_info);
		vector<int> outputLabel2 = nn_c.apply(testInput, print_info);

		cout << endl << "Predicition before training | Predicition after training | true labels: " << endl;
		for (int i = 0; i < 100; i++)
			cout << outputLabel1[i];
		cout << endl;
		for (int i = 0; i < 100; i++)
			cout << outputLabel2[i];
		cout << endl;
		for (int i = 0; i < 100; i++)
			cout << testLabel[i];

		cout << endl << "Difference to true labels -> before training | after training: " << endl;
		for (int i = 0; i < 100; i++)
			cout << abs(outputLabel1[i] - testLabel[i]);
		cout << endl;
		for (int i = 0; i < 100; i++)
			cout << abs(outputLabel2[i] - testLabel[i]);
		cout << endl;

		// nn_c.train(numEpoch, trainInput, trainLabel, testInput, testLabel, print_info);
		// cout << "Training " << numEpoch << " epoches takes " << (float)(clock()-time) / CLOCKS_PER_SEC << " seconds." << endl;

		cout << "Train Classifier Completed." << endl;
	}
	return 0;
}
