#ifndef RNG_H
#define RNG_H

#include <random>
#include "batchData.h"

extern random_device rd;
extern mt19937 gen;
normal_distribution<float> stdNorm(0,1);
uniform_real_distribution<float> stdUnif(0,1);
exponential_distribution<float> stdExp(1);

template<typename T>
vector<batchData<T>> randomGenerator(string dist_name, size_t numBatch, size_t batchSize)
{
    vector<batchData<T>> ans(numBatch, batchData<T>(batchSize,0));
    if (dist_name == "gaussian")
        for (int i = 0; i < numBatch; i++)
            for (int b = 0; b < batchSize; b++)
                ans[i][b] = stdNorm(gen);
    else if (dist_name == "uniform")
        for (int i = 0; i < numBatch; i++)
            for (int b = 0; b < batchSize; b++)
                ans[i][b] = stdUnif(gen);
    else if (dist_name == "exp")
        for (int i = 0; i < numBatch; i++)
            for (int b = 0; b < batchSize; b++)
                ans[i][b] = stdExp(gen);
    return ans;
}

#endif
