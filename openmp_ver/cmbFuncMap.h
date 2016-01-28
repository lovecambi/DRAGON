#ifndef CMBFUNCMAP_H
#define CMBFUNCMAP_H

#include <map>
#include "batchData.h"

template<typename R>
using cmb_func_type = vector<R> (*) (vector<vector<R>*>);

// Normal distribution combine function
// Inputs: mean - mu, log(variance) - log_sigma2, random numbers obey N(0,1) - epsilon
// Output: mu + exp(0.5*log_sigma2) .* epsilon (under Matlab syntax)
template <typename T>
vector<batchData<T>> NormRnd(vector<vector<batchData<T>>*> inputDataVec)
{
    // Guarantee number of input parameter is correct
    const int numInput = 3;
    assert(inputDataVec.size() == numInput);
    // Guarantee input dimensions are correct;
    assert( (*inputDataVec[0]).size() == (*inputDataVec[1]).size() );
    assert( (*inputDataVec[0]).size() == (*inputDataVec[2]).size() );
    // Compute output
    size_t dim = (*inputDataVec[0]).size();
    vector<batchData<T>> * p_mu = inputDataVec[0];
    vector<batchData<T>> * p_log_sigma2 = inputDataVec[1];
    vector<batchData<T>> * p_epsilon = inputDataVec[2];
    vector<batchData<T>> ans(dim);
    
    batchData<T> temp;
#pragma omp parallel for private(temp)
    for (int i = 0; i < dim; i++) {
        temp = 0.5 * (*p_log_sigma2)[i];
        ans[i] = (*p_mu)[i] + Exp(temp) * (*p_epsilon)[i];
    }
    return ans;
}

// Partial derivative of normal distribution combine function w.r.t. mu
// Inputs: mean - mu, log(variance) - log_sigma2, random numbers obey N(0,1) - epsilon
// Output: all one vector
template <typename T>
vector<batchData<T>> NormRnd_drv_mu(vector<vector<batchData<T>>*> inputDataVec)
{
    // Guarantee number of input parameter is correct
    const int numInput = 3;
    assert(inputDataVec.size() == numInput);
    // Guarantee input dimensions are correct;
    assert( (*inputDataVec[0]).size() == (*inputDataVec[1]).size() );
    assert( (*inputDataVec[0]).size() == (*inputDataVec[2]).size() );
    // Compute output
    size_t dim = (*inputDataVec[0]).size();
    size_t batchSize = (*inputDataVec[0])[0].size();
    return vector<batchData<T>>(dim, batchData<T>(batchSize,T(1)));
}

// Partial derivative of normal distribution combine function w.r.t. log_sigma2
// Inputs: mean - mu, log(variance) - log_sigma2, random numbers obey N(0,1) - epsilon
// Output: 0.5 * epsilon * exp(0.5*log_sigma2)
template <typename T>
vector<batchData<T>> NormRnd_drv_log_sigma2(vector<vector<batchData<T>>*> inputDataVec)
{
    // Guarantee number of input parameter is correct
    const int numInput = 3;
    assert(inputDataVec.size() == numInput);
    // Guarantee input dimensions are correct;
    assert( (*inputDataVec[0]).size() == (*inputDataVec[1]).size() );
    assert( (*inputDataVec[0]).size() == (*inputDataVec[2]).size() );
    // Compute output
    size_t dim = (*inputDataVec[0]).size();
    vector<batchData<T>> * p_log_sigma2 = inputDataVec[1];
    vector<batchData<T>> * p_epsilon = inputDataVec[2];
    vector<batchData<T>> ans(dim);
    
    batchData<T> temp;
    for (int i = 0; i < dim; i++) {
        temp = 0.5 * (*p_log_sigma2)[i];
        ans[i] = 0.5 * (*p_epsilon)[i] * Exp(temp);
    }
    return ans;
}

// Struct to store function pointer of combine function and its partial derivatives
template<typename T>
struct cmbFuncInfo {
    cmb_func_type<batchData<T>> func;
    vector<cmb_func_type<batchData<T>>> drv;
    vector<vector<batchData<T>>*> inputPtr;
    cmbFuncInfo (cmb_func_type<batchData<T>> _func = NULL, vector<cmb_func_type<batchData<T>>> _drv = vector<cmb_func_type<batchData<T>>>()) : func(_func), drv(_drv) {}
    cmbFuncInfo& operator=(const cmbFuncInfo& other) {
        if (this != &other) {
            func = other.func;
            drv = other.drv;
            inputPtr = other.inputPtr;
        }
        return *this;
    }
};

template<typename T>
class cmbFunctionMap {
public:
    map<string, cmbFuncInfo<T>> funcList;
    map<cmb_func_type<batchData<T>>, string> funcNameList;
public:
    // Constructor: add all existing function to the map
    cmbFunctionMap()
    {
        // Use name to find functions
        funcList["null"] = cmbFuncInfo<T>(NULL, vector<cmb_func_type<batchData<T>>>());
        funcList["NormRnd"] = cmbFuncInfo<T>(NormRnd,{NormRnd_drv_mu, NormRnd_drv_log_sigma2, NULL});
        // Use function point to find its name
#ifdef __APPLE__
        funcNameList[NormRnd] = "NormRnd";
        funcNameList[NULL] = "null";
#endif
    }
    
    // Destructor
    ~cmbFunctionMap()
    {
        funcList.clear();
        funcNameList.clear();
    }
    
    // [] overload: given name return the function and derivative
    cmbFuncInfo<T> operator[](string name)
    {
        return funcList[name];
    }
    
    // () overload: given function return its name
    string operator()(cmb_func_type<batchData<T>> func)
    {
        return funcNameList[func];
    }
};

#endif
