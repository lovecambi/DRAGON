#ifndef TRSFUNCMAP_H
#define TRSFUNCMAP_H

#include <map>
#include "batchData.h"

template<typename R>
using trs_func_type = R (*) (const R&);

/* Elementary functions used in objective functions */

// log function
template <typename T>
batchData<T> Log(const batchData<T>& input)
{
    size_t batchSize = input.size();
    batchData<T> ans(batchSize);
    for (int i = 0; i < batchSize; i++)
        ans[i] = log(input[i]);
    return ans;
}

// log function
template <typename T>
batchData<T> Exp(const batchData<T>& input)
{
    size_t batchSize = input.size();
    batchData<T> ans(batchSize);
    for (int i = 0; i < batchSize; i++)
        ans[i] = exp(input[i]);
    return ans;
}

/* Transfunctions */

// tanh function
template <typename T>
batchData<T> Tanh(const batchData<T>& input)
{
    size_t batchSize = input.size();
    batchData<T> ans(batchSize);
    for (int i = 0; i < batchSize; i++)
        ans[i] = tanh(input[i]);
    return ans;
}

// sigmoid function
template <typename T>
batchData<T> Sigmoid(const batchData<T>& input)
{
    batchData<T> tmp = input * 0.5;
    return (Tanh(tmp) + 1) * 0.5;
}

// identity function
template <typename T>
batchData<T> Identity(const batchData<T>& input)
{
    return input * 1;
}

// derivative tanh function
template <typename T>
batchData<T> Tanh_drv(const batchData<T>& input)
{
    batchData<T> tmp = Tanh(input);
    return 1 - tmp * tmp;
}

// derive identity function
template <typename T>
batchData<T> Identity_drv(const batchData<T>& input)
{
    return batchData<T>(input.size(), T(1));
}

// derivative sigmoid function
template <typename T>
batchData<T> Sigmoid_drv(const batchData<T>& input)
{
    batchData<T> tmp = Sigmoid(input);
    return  tmp * (1 - tmp);
}

template<typename T>
class trsFunctionMap {
public:
    map<string, vector<trs_func_type<batchData<T>>>> funcList;
    map<trs_func_type<batchData<T>>, string> funcNameList;
public:
    // Constructor: add all existing function to the map
    trsFunctionMap()
    {
        // Use name to find functions
        funcList["identity"] = {Identity<T>, Identity_drv<T>};
        funcList["tanh"] = {Tanh<T>, Tanh_drv<T>};
        funcList["sigmoid"] = {Sigmoid<T>, Sigmoid_drv<T>};
        // Use function point to find its name
#ifdef __APPLE__
        funcNameList[Identity] = "identity";
        funcNameList[Tanh] = "tanh";
        funcNameList[Sigmoid] = "sigmoid";
#endif
    }
    
    // Destructor
    ~trsFunctionMap()
    {
        funcList.clear();
        funcNameList.clear();
    }
    
    // [] overload: given name return the function and derivative
    vector<trs_func_type<batchData<T>>> operator[](string name)
    {
        return funcList[name];
    }
    
    // () overload: given function return its name
    string operator()(trs_func_type<batchData<T>> func)
    {
        return funcNameList[func];
    }
};

#endif
