#ifndef OBJFUNCMAP_H
#define OBJFUNCMAP_H

#include <map>
#include <vector>
#include <string>
#include <cmath>

#define TINY 0.000001

typedef float (*obj_func_type) (const float&, const float&);

struct obj_functor
{ 
    obj_func_type func;
    obj_functor(obj_func_type _func = NULL) : func(_func) {} 
    __host__ __device__ float operator()(const float& t, const float& y) const {
        return (*func)(t,y);
    }
    bool operator<(const obj_functor& rhs) const {
      return rhs.func < this->func;
    }
    obj_functor& operator=(const obj_functor& other) {
        if (this != &other) {
            func = other.func;
        }
        return *this;
    }
};


// Cross entropy objective function
__device__ float CrossEntropy(const float& t, const float& y) { 
    // Avoid numerical issue of divide by zero
    float temp_y = (y < TINY) ? TINY : ( (y > 1-TINY) ? 1-TINY : y );
    return - t * log(temp_y) - (1-t) * log(1-temp_y); 
}

// Square error objective function
__device__ float SquareError(const float& t, const float& y) { 
    return (t-y) * (t-y) / 2; 
}

// derivative Cross entropy objective function
__device__ float CrossEntropy_drv(const float& t, const float& y) { 
    // Avoid numerical issue of divide by zero
    float temp_y = (y < TINY) ? TINY : ( (y > 1-TINY) ? 1-TINY : y );
    return (1-t)/(1-temp_y) - t/temp_y; 
}

// derivative Square error objective function
__device__ float SquareError_drv(const float& t, const float& y) { 
    return y - t;
}

struct objFuncInfo {
    obj_functor func;
    obj_functor drv;
    objFuncInfo (obj_functor _func = obj_functor(NULL), obj_functor _drv = obj_functor(NULL)) : func(_func), drv(_drv) {}
    objFuncInfo& operator=(const objFuncInfo& other) {
        if (this != &other) {
            func = other.func;
            drv = other.drv;
        }
        return *this;
    }
};

// Get device function pointer through a kernel, the order should be same as the map func_ptr_idx
__global__ void get_address_of_device_obj_function(obj_func_type *ptr_to_ptr) 
{ 
  ptr_to_ptr[0] = &CrossEntropy; 
  ptr_to_ptr[1] = &CrossEntropy_drv;
  ptr_to_ptr[2] = &SquareError; 
  ptr_to_ptr[3] = &SquareError_drv; 
} 

class objFunctionMap {
public:
    map<string, objFuncInfo > funcList;
    map<obj_functor, string> funcNameList;
public:
    // Constructor: add all existing function to the map
    objFunctionMap()
    {
        map<string, int> func_ptr_idx;
        func_ptr_idx["CrossEntropy"] = 0;
        func_ptr_idx["CrossEntropy_drv"] = 1;
        func_ptr_idx["SquareError"] = 2;
        func_ptr_idx["SquareError_drv"] = 3;

        thrust::device_vector<obj_func_type> func_ptr(func_ptr_idx.size()); 
        get_address_of_device_obj_function<<<1,1>>>(raw_pointer_cast(func_ptr.data())); 

        // Use name to find functions
        funcList["CrossEntropy"] = objFuncInfo(obj_functor(func_ptr[func_ptr_idx["CrossEntropy"]]), obj_functor(func_ptr[func_ptr_idx["CrossEntropy_drv"]]));
        funcList["SquareError"] = objFuncInfo(obj_functor(func_ptr[func_ptr_idx["SquareError"]]), obj_functor(func_ptr[func_ptr_idx["SquareError_drv"]]));
        // Use function point to find its name
        funcNameList[obj_functor(func_ptr[func_ptr_idx["CrossEntropy"]])] = "CrossEntropy";
        funcNameList[obj_functor(func_ptr[func_ptr_idx["SquareError"]])] = "SquareError";
    }
    
    // Destructor
    ~objFunctionMap()
    {
        funcList.clear();
        funcNameList.clear();
    }
    
    // [] overload: given name return the function and derivative
    objFuncInfo operator[](string name)
    {
        return funcList[name];
    }
    
    // () overload: given function return its name
    string operator()(obj_functor func)
    {
        return funcNameList[func];
    }
};

#endif
