#ifndef floatRSFUNCMAP_H
#define floatRSFUNCMAP_H

#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <thrust/device_vector.h>

typedef float (*trs_func_type) (const float&);

struct trs_functor
{ 
    trs_func_type func;
    trs_functor(trs_func_type _func = NULL) : func(_func) {} 
    __host__ __device__ float operator()(const float& x) const {
        return (*func)(x);
    }
    bool operator<(const trs_functor& rhs) const {
      return rhs.func < this->func;
    }
    trs_functor& operator=(const trs_functor& other) {
        if (this != &other) {
            func = other.func;
        }
        return *this;
    }
};

/* floatransfer functions */

// tanh function
__device__ float Tanh(const float& x) { 
    return tanh(x); 
}

// sigmoid function
__device__ float Sigmoid(const float& x) { 
    return (tanh(x/2) + 1) / 2;
}

// identity function
__device__ float Identity(const float& x) { 
    return x;
}

// derivative tanh function
__device__ float Tanh_drv(const float& x) { 
    float temp = tanh(x);
    return 1 - temp * temp;
}

// derivative sigmoid function
__device__ float Sigmoid_drv(const float& x) { 
    float temp = 1 / ( 1 + exp(-x) );        
    return temp * (1-temp);
}

// derivative identity function
__device__ float Identity_drv(const float& x) { 
    return 1;
}

struct trsFuncInfo {
    trs_functor func;
    trs_functor drv;
    trsFuncInfo (trs_functor _func = trs_functor(NULL), trs_functor _drv = trs_functor(NULL)) : func(_func), drv(_drv) {}
    trsFuncInfo& operator=(const trsFuncInfo& other) {
        if (this != &other) {
            func = other.func;
            drv = other.drv;
        }
        return *this;
    }
};

// Get device function pointer through a kernel, the order should be same as the map func_ptr_idx
__global__ void get_address_of_device_trs_function(trs_func_type *ptr_to_ptr) 
{ 
  ptr_to_ptr[0] = &Identity; 
  ptr_to_ptr[1] = &Identity_drv;
  ptr_to_ptr[2] = &Tanh; 
  ptr_to_ptr[3] = &Tanh_drv; 
  ptr_to_ptr[4] = &Sigmoid; 
  ptr_to_ptr[5] = &Sigmoid_drv; 
} 

class trsFunctionMap {
public:
    map<string, trsFuncInfo> funcList;
    map<trs_functor, string> funcNameList;
public:
    // Constructor: add all existing function to the map
    trsFunctionMap()
    {
        map<string, int> func_ptr_idx;
        func_ptr_idx["identity"] = 0;
        func_ptr_idx["identity_drv"] = 1;
        func_ptr_idx["tanh"] = 2;
        func_ptr_idx["tanh_drv"] = 3;
        func_ptr_idx["sigmoid"] = 4;
        func_ptr_idx["sigmoid_drv"] = 5;

        thrust::device_vector<trs_func_type> func_ptr(func_ptr_idx.size()); 
        get_address_of_device_trs_function<<<1,1>>>(raw_pointer_cast(func_ptr.data())); 

        // Use name to find functions
        funcList["identity"] = trsFuncInfo(trs_functor(func_ptr[func_ptr_idx["identity"]]), trs_functor(func_ptr[func_ptr_idx["identity_drv"]]));
        funcList["tanh"] = trsFuncInfo(trs_functor(func_ptr[func_ptr_idx["tanh"]]), trs_functor(func_ptr[func_ptr_idx["tanh_drv"]]));
        funcList["sigmoid"] = trsFuncInfo(trs_functor(func_ptr[func_ptr_idx["sigmoid"]]), trs_functor(func_ptr[func_ptr_idx["sigmoid_drv"]]));
        // Use function point to find its name
        funcNameList[trs_functor(func_ptr[func_ptr_idx["identity"]])] = "identity";
        funcNameList[trs_functor(func_ptr[func_ptr_idx["tanh"]])] = "tanh";
        funcNameList[trs_functor(func_ptr[func_ptr_idx["sigmoid"]])] = "sigmoid";
    }
    
    // Destructor
    ~trsFunctionMap()
    {
        funcList.clear();
        funcNameList.clear();
    }
    
    // [] overload: given name return the function and derivative
    trsFuncInfo operator[](string name)
    {
        return funcList[name];
    }
    
    // () overload: given function return its name
    string operator()(trs_functor func)
    {
        return funcNameList[func];
    }
};

#endif
