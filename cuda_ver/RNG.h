#ifndef RNG_H
#define RNG_H

#include <thrust/device_vector.h>
#include <curand.h>

class RNG
{
public:
    curandGenerator_t prng;
public:
    RNG()
    {
        // Create a pseudo-random number generator
        curandGenerator_t prng;
        curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
        // Set the seed for the random number generator using the system clock
        curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
    }
    void randomGenerator(thrust::device_vector<float>& rand_vec, string dist_name, size_t numBatch, size_t batchSize)
    {
        
        if (dist_name == "gaussian")
            curandGenerateNormal(prng, thrust::raw_pointer_cast(&rand_vec[0]), numBatch * batchSize, 0, 1);
        else if (dist_name == "uniform")
            curandGenerateUniform(prng, thrust::raw_pointer_cast(&rand_vec[0]), numBatch * batchSize);
    }
};





#endif
