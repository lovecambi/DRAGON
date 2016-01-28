#ifndef OBJFUNCMAP_H
#define OBJFUNCMAP_H

#include <map>
#include "trsFuncMap.h"
#include "batchData.h"

template<typename T>
using obj_func_type = T (*) (const vector<batchData<T>>&, const vector<batchData<T>>&);

template<typename T>
using obj_func_drv_type = batchData<T> (*) (const batchData<T>&, const batchData<T>&);

// Cross entropy objective function
template<typename T>
T CrossEntropy(const vector<batchData<T>>& targets, const vector<batchData<T>>& outputs)
{
    assert( targets.size() == outputs.size() );
    size_t dim = targets.size();
    T ans = 0;
    for (int i = 0; i < dim; i++) {
        batchData<T> tmp = -outputs[i] + 1;
        batchData<T> part_ans = targets[i] * Log(outputs[i]) + (1-targets[i]) * Log(tmp);
        ans -= part_ans.sum();
    }
    return ans;
}

// Partial derivative of cross entropy w.r.t. outputs
template<typename T>
batchData<T> CrossEntropy_drv(const batchData<T>& target, const batchData<T>& output)
{
    assert( target.size() == output.size() );
    return (1 - target) / (1 - output) - target / output;
}


// Struct to store function pointer of objective function and partial derivatives w.r.t. output
// Since it is not necessary to compute partial derivatives w.r.t. target
template<typename T>
struct objFuncInfo {
    obj_func_type<T> func;
    obj_func_drv_type<T> drv;
    objFuncInfo (obj_func_type<T> _func = NULL, obj_func_drv_type<T> _drv = NULL) : func(_func), drv(_drv) {}
    objFuncInfo& operator=(const objFuncInfo& other) {
        if (this != &other) {
            func = other.func;
            drv = other.drv;
        }
        return *this;
    }
};

template<typename T>
class objFunctionMap {
public:
    map<string, objFuncInfo<T>> funcList;
    map<obj_func_type<T>, string> funcNameList;
public:
    // Constructor: add all existing function to the map
    objFunctionMap()
    {
        // Use name to find functions
        funcList["CrossEntropy"] = objFuncInfo<T>(CrossEntropy, CrossEntropy_drv);
        // Use function point to find its name
#ifdef __APPLE__
        funcNameList[CrossEntropy] = "CrossEntropy";
#endif
    }
    
    // Destructor
    ~objFunctionMap()
    {
        funcList.clear();
        funcNameList.clear();
    }
    
    // [] overload: given name return the function and derivative
    objFuncInfo<T> operator[](string name)
    {
        return funcList[name];
    }
    
    // () overload: given function return its name
    string operator()(obj_func_type<T> func)
    {
        return funcNameList[func];
    }
};

#endif
