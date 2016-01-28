#ifndef NODE_
#define NODE_

#include <random>
#include "batchData.h"

#define EPS 0.000001

random_device rd;
mt19937 gen(rd());
normal_distribution<float> normrnd(0,0.1);

template <typename T>
class node {
public:
	int numInputs;			// number of adjacent nodes from previous layer
	vector<T> weights;		// weights of corresponding adjacent nodes from previous layer
	vector<T> d_weights;	// update incrementing value of incoming weights
	vector<T> cum_weights;	// cumulative square sum of weights, used to compute adaptive step length
	T bias;					// bias of this node
	T d_bias;				// update incrementing value of bias
	T cum_bias;				// cumulative square sum of bias, used to compute adaptive step length
public:
	// Constructor
	node(int _numInputs = 0) : numInputs(_numInputs)
    {
		weights = vector<T>(numInputs,0);
		d_weights = vector<T>(numInputs,0);
		cum_weights = vector<T>(numInputs,EPS);
		// Initialize the weights are random variables
		for (int i = 0; i < numInputs; i++)
			weights[i] = normrnd(gen);
		bias = 0;
		d_bias = 0;
		cum_bias = EPS;
	}

	// Destructor
	virtual ~node()
    {
		vector<T>().swap(weights);
		vector<T>().swap(d_weights);
		vector<T>().swap(cum_weights);
	}
    
    // = overload
    node<T>& operator=(const node<T>& other) {
        if (this != &other) {
            numInputs = other.numInputs;
            weights = other.weights;
            d_weights = other.d_weights;
            cum_weights = other.cum_weights;
            bias = other.bias;
            d_bias = other.d_bias;
            cum_bias = other.cum_bias;
        }
        return *this;
    }

	// Assign input weights for the node
	void setParams(vector<T> _weights, T _bias)
    {
		weights = _weights;
		bias = _bias;
	}

	// Assign derivative of weights for the node
	void setDParams(vector<T> _d_weights, T _d_bias)
    {
			d_weights = _d_weights;
			d_bias = _d_bias;
	}

	// Compute the activation given input
	batchData<T> computeLinearOutput(const vector<batchData<T>>& inputs)
    {
		batchData<T> ans = inputs[0] * weights[0];
		for (int i = 1; i < numInputs; i++)
			ans += inputs[i] * weights[i];
		ans = ans + bias;
		return ans;
	}

	// Use adaptive gradient step size to update parameter
	void adaGradUpdateParameter(float epsilon)
	{
		for (int i = 0; i < numInputs; i++)
		{
			cum_weights[i] += d_weights[i] * d_weights[i];
            weights[i] -= epsilon * d_weights[i] / sqrt(cum_weights[i]);
		}
		cum_bias += d_bias * d_bias;
        bias -= epsilon * d_bias / sqrt(cum_bias);
	}
};

#endif