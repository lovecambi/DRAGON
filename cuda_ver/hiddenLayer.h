#ifndef HIDDENLAYER_
#define HIDDENLAYER_

#include "trsFuncMap.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/extrema.h>
#include <curand.h>
#include <cublas_v2.h>

#define EPS 0.000001
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

void printMatrix(const thrust::device_vector<float>& mtx, size_t num_row, size_t num_col)
{
    // assert( mtx.size() == num_row * num_col );
    for (int i = 0; i < num_row; i++)
    {
        for (int j = 0; j < num_col; j++)
            cout << mtx[IDX2C(i,j,num_row)] << " ";
        cout << endl;
    }
}

// Auxiliary functor to compute adaptive gradient
struct d_cum_update_func
{ 
    d_cum_update_func()   {}
    template <typename Tuple>
    __host__ __device__ void operator()(Tuple t) const {
        // Tuple: (param_drv, d_param, cum_param)
        // cum_param = param_drv * param_drv + cum_param;
        thrust::get<2>(t) = thrust::get<0>(t) * thrust::get<0>(t) + thrust::get<2>(t);
        // d_param = param_drv / sqrt(cum_param);
        thrust::get<1>(t) = thrust::get<0>(t) / sqrt(thrust::get<2>(t));
    }
};

// Actual gradient descent functor
struct update_func
{
	float epsilon;
	update_func(float _epsilon) : epsilon(_epsilon) {}
	__host__ __device__ float operator()(const float& d_param, const float& param) const {
        return param - epsilon * d_param;
    }
};

class hiddenLayer {
public:
	int numInputs;                          	// number of input value, i.e. number of nodes in previous layer
	int numNodes;                           	// number of nodes in this layer

	int batchSize;								// batch size of batch training
	cublasHandle_t handle;						// cuBlas handle for matrix operation
	trsFuncInfo trsFunc;       				    // transfer function and its derivative between activation and output of all nodes

	thrust::device_vector<float> weights;		// weights of the hidden layer, i.e. size (numNodes x numInputs) matrix W stored in row order
	thrust::device_vector<float> bias;			// bias of the hidden layer, i.e. size numNodes vector
	thrust::device_vector<float> d_weights;		// partial derivative of the weights
	thrust::device_vector<float> d_bias;		// partial derivative of the bias
	thrust::device_vector<float> cum_weights;	// cummulative square sum of weights, used to compute adaptive step length
	thrust::device_vector<float> cum_bias;		// cummulative square sum of bias, used to compute adaptive step length

	thrust::device_vector<float> one_vec;		// auxiliary vector of all ones
	thrust::device_vector<float> linearOutput;	// activation of the layer, i.e. size (numNodes x batchSize) matrix stored in row order

public:
    // Constructor
    hiddenLayer(int _numInputs, int _numNodes, int _batchSize, trsFuncInfo _trsFuncInfo) \
    : numInputs(_numInputs), numNodes(_numNodes), batchSize(_batchSize), trsFunc(_trsFuncInfo)
    {
		// Create a handle for CUBLAS
		cublasCreate(&handle);
    	initialize();
    }

    // Constructor with vector of trs_func_type
    hiddenLayer(int _numInputs = 1, int _numNodes = 1, int _batchSize = 1, string trsFuncName = "identity") \
    : numInputs(_numInputs), numNodes(_numNodes), batchSize(_batchSize)
    {
    	trsFunctionMap trsFuncMap;
    	trsFunc = trsFuncMap[trsFuncName];
		// Create a handle for CUBLAS
		cublasCreate(&handle);
    	initialize();
    }

	// Destructor
	virtual ~hiddenLayer()
    {
    	// Destroy the handle
		cublasDestroy(handle);
    	// device_vector are automatically released by cudaFree()
	}
    
    // = overload
    hiddenLayer& operator=(const hiddenLayer& other) {
        if (this != &other) {
            numInputs = other.numInputs;
            numNodes = other.numNodes;

            batchSize = other.batchSize;
            trsFunc = other.trsFunc;

            weights = other.weights;
            bias = other.bias;
            d_weights = other.d_weights;
            d_bias = other.d_bias;
            cum_weights = other.cum_weights;
            cum_bias = other.cum_bias;

            one_vec = other.one_vec;
            linearOutput = other.linearOutput;
        }
        return *this;
    }

	// Assign weights for incoming edges of all nodes
	void setParams(thrust::device_vector<float>& _weights, thrust::device_vector<float>& _bias)
    {
		weights = _weights;
		bias = _bias;
	}

	// Assign deriabative weights for incoming edges of all nodes
	void setDParams(thrust::device_vector<float>& weights_drv, thrust::device_vector<float>& bias_drv)
    {
        // cum_param += param_drv^2, d_param = param_drv / sqrt(cum_param)
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(weights_drv.begin(), d_weights.begin(), cum_weights.begin())),
             thrust::make_zip_iterator(thrust::make_tuple(weights_drv.end(), d_weights.end(), cum_weights.end())),
             d_cum_update_func());
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(bias_drv.begin(), d_bias.begin(), cum_bias.begin())),
             thrust::make_zip_iterator(thrust::make_tuple(bias_drv.end(), d_bias.end(), cum_bias.end())),
             d_cum_update_func());
	}

	// Compute the output of this hidden layer
	void computeOutputs(thrust::device_vector<float>& inputs, thrust::device_vector<float>& outputs)
    {
    	assert(inputs.size() == numInputs * batchSize);
    	assert(outputs.size() == numNodes * batchSize);
    	// Compute linear outputs: alpha = W * gamma + b * vec(1)^T
	 	gpu_blas_mmul(bias, one_vec, linearOutput, 
	 		numNodes, batchSize, 1,
	 		1, 0, 
	 		CUBLAS_OP_N, CUBLAS_OP_T); 
	 	gpu_blas_mmul(weights, inputs, linearOutput, 
	 		numNodes, batchSize, numInputs, 
	 		1, 1, 
	 		CUBLAS_OP_N, CUBLAS_OP_N); 
	 	// Compute outputs by passing linear outputs through transfer function: beta = f(alpha)
	 	thrust::transform(linearOutput.begin(), linearOutput.end(), 
	 		outputs.begin(), trsFunc.func);
	}
	
	// Compute the output of this hidden layer
	void computeOutputs(thrust::device_vector<float>& inputs, thrust::device_vector<float>& outputs, thrust::device_vector<float>& outputs_drv)
    {
    	assert(inputs.size() == numInputs * batchSize);
    	assert(outputs.size() == numNodes * batchSize);
    	// Compute linear outputs: alpha = W * gamma + b * vec(1)^T
	 	gpu_blas_mmul(bias, one_vec, linearOutput, 
	 		numNodes, batchSize, 1,
	 		1, 0, 
	 		CUBLAS_OP_N, CUBLAS_OP_N); 
	 	gpu_blas_mmul(weights, inputs, linearOutput, 
	 		numNodes, batchSize, numInputs, 
	 		1, 1, 
	 		CUBLAS_OP_N, CUBLAS_OP_N); 
	 	// Compute outputs by passing linear outputs through transfer function: beta = f(alpha)
	 	thrust::transform(linearOutput.begin(), linearOutput.end(), 
	 		outputs.begin(), trsFunc.func);
	 	// Compute derivative of transfer function: sigma^prime(alpha)
	 	thrust::transform(linearOutput.begin(), linearOutput.end(), 
	 		outputs_drv.begin(), trsFunc.drv);
	}

	// Update parameter
	void updateParameter(float epsilon)
	{
		// param -= epsilon * d_param
	 	thrust::transform(d_weights.begin(), d_weights.end(), 
	 		weights.begin(), 
	 		weights.begin(), update_func(epsilon));
	 	thrust::transform(d_bias.begin(), d_bias.end(), 
	 		bias.begin(), 
	 		bias.begin(), update_func(epsilon));
	}

	// << overload: print the vector for debug
    friend ostream& operator<<(ostream& outfile, hiddenLayer& hl)
    {
        trsFunctionMap trsFuncMap;
        // Save input dimension, node number, batch size and transfer function name
        outfile << "{ " << endl;
        outfile << "\t" << hl.numInputs << " " << hl.numNodes << " ";
        outfile << hl.batchSize << " " << trsFuncMap(hl.trsFunc.func) << endl;
        // Save detailed settings for a hidden layer into host_vector
        thrust::host_vector<float> temp_weights = hl.weights;
        thrust::host_vector<float> temp_bias = hl.bias;
        thrust::host_vector<float> temp_cum_weights = hl.cum_weights;
        thrust::host_vector<float> temp_cum_bias = hl.cum_bias;
		for (int i = 0; i < hl.numNodes; i++) {
            outfile << "\t[ " << endl;
            // Save bias and cum_bias of the node
            outfile << "\t\t" << temp_bias[i] << " " << temp_cum_bias[i] << endl << "\t\t";
            // Save weights of the node
            for (int k = 0; k < hl.numInputs; k++)
                outfile << temp_weights[IDX2C(i,k,hl.numNodes)] << " ";
            outfile << endl << "\t\t";
            // Save cum_weights of the node
            for (int k = 0; k < hl.numInputs; k++)
                outfile << temp_cum_weights[IDX2C(i,k,hl.numNodes)] << " ";
            outfile << endl << "\t];" << endl;
        }
        outfile << "};" << endl << endl;
        return outfile;
    }

    // >> overload: load data from istream
    friend istream& operator>>(istream& infile, hiddenLayer& hl)
    {
        string temp, func_name;
        trsFunctionMap trsFuncMap;
    	infile >> temp;     // Jump line "{"
        // Load general parameters for a hidden layer
        infile >> hl.numInputs >> hl.numNodes >> hl.batchSize >> func_name;
        hl.trsFunc = trsFuncMap[func_name];
        // Allocate memory for current layer to store information of nodes
        hl.initialize();
        // Save detailed settings for a hidden layer into host_vector
        thrust::host_vector<float> temp_weights(hl.numNodes*hl.numInputs,0);
    	thrust::host_vector<float> temp_bias(hl.numNodes,0);
        thrust::host_vector<float> temp_cum_weights(hl.numNodes*hl.numInputs,0);
    	thrust::host_vector<float> temp_cum_bias(hl.numNodes,0);
        for (int i = 0; i < hl.numNodes; i++) {
            infile >> temp;     // Jump line "    ["
            // Save numInputs, bias and cum_bias of the node
            infile >> temp_bias[i] >> temp_cum_bias[i];
            // Load weights of the node
            for (int k = 0; k < hl.numInputs; k++)
                infile >> temp_weights[IDX2C(i,k,hl.numNodes)];
            // Load cum_weights of the node
            for (int k = 0; k < hl.numInputs; k++)
                infile >> temp_cum_weights[IDX2C(i,k,hl.numNodes)];
            infile >> temp;  // Jump line "     ];"
        }
        infile >> temp;      // Jump line "};" and the following empty line
        // Copy host_vector to device_vector
        thrust::copy(temp_weights.begin(), temp_weights.end(), hl.weights.begin());
        thrust::copy(temp_bias.begin(), temp_bias.end(), hl.bias.begin());
        thrust::copy(temp_cum_weights.begin(), temp_cum_weights.end(), hl.cum_weights.begin());
        thrust::copy(temp_cum_bias.begin(), temp_cum_bias.end(), hl.cum_bias.begin());
    	return infile;
    }
private:
	// Initialize the hidden layer, alloc device memory and assign initial value
	void initialize()
	{
		// Allocate device memory and initialized for weights and bias
    	weights = thrust::device_vector<float>(numNodes*numInputs,0);
    	gpu_fill_norm_rand(weights, numNodes, numInputs);
    	bias = thrust::device_vector<float>(numNodes,0);
    	// Allocate device memory and initialized for derivative of weights and bias
    	d_weights = thrust::device_vector<float>(numNodes*numInputs,0);
    	d_bias = thrust::device_vector<float>(numNodes,0);
    	// Allocate device memory and initialized for cummulative weights and cummulative bias
    	cum_weights = thrust::device_vector<float>(numNodes*numInputs, EPS);
    	cum_bias = thrust::device_vector<float>(numNodes, EPS);
    	// Allocate device memory and initialized for linear output
    	linearOutput = thrust::device_vector<float>(numNodes*batchSize,0);
    	// Allocate device memory for auxiliary all one vector
    	one_vec = thrust::device_vector<float>(batchSize,1);
	}

	// Fill the array A(num_row, num_col) with normal random numbers on GPU
	void gpu_fill_norm_rand(thrust::device_vector<float>& A, int num_row, int num_col, float stddev = 0.0001f, float mean = 0.f) 
	{
		// Create a pseudo-random number generator
		curandGenerator_t prng;
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
		// Set the seed for the random number generator using the system clock
		curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
		// Fill the array with random numbers on the device
		curandGenerateNormal(prng, thrust::raw_pointer_cast(&A[0]), num_row * num_col, mean, stddev);
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
};

#endif
