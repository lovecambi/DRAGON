#ifndef CMBFUNCMAP_H
#define CMBFUNCMAP_H

#include <map>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

#define CMB_MAX_ARG_NUM 5
typedef void (*cmb_func_type) (float&, const float&, const float&, const float&, const float&, const float&);

struct cmb_functor
{ 
    cmb_func_type func;
    cmb_functor(cmb_func_type _func = NULL) : func(_func) {} 
    template <typename Tuple>
    __host__ __device__ void operator()(Tuple t) const {
        (*func)(thrust::get<0>(t),thrust::get<1>(t),thrust::get<2>(t),thrust::get<3>(t),thrust::get<4>(t),thrust::get<5>(t));
    }
    bool operator<(const cmb_functor& rhs) const {
      return rhs.func < this->func;
    }
    bool isNull()   { return func == NULL; }
    cmb_functor& operator=(const cmb_functor& other) {
        if (this != &other) {
            func = other.func;
        }
        return *this;
    }
};

// Normal distribution combine function
// Inputs: mean - mu, log(variance) - log_sigma2, random numbers obey N(0,1) - epsilon
// Output: mu + exp(0.5*log_sigma2) .* epsilon (under Matlab syntax)
__device__ void NormRnd(float& output, const float& mu, const float& log_sigma2, const float& epsilon, const float& dummy1 = 0, const float& dummy2 = 0) {
    output = mu + exp( 0.5 * log_sigma2 ) * epsilon;
}

// Partial derivative of normal distribution combine function w.r.t. mu
// Inputs: mean - mu, log(variance) - log_sigma2, random numbers obey N(0,1) - epsilon
// Output: all one vector
__device__ void NormRnd_drv_mu(float& output, const float& mu, const float& log_sigma2, const float& epsilon, const float& dummy1 = 0, const float& dummy2 = 0) {
    output = 1;
}

// Partial derivative of normal distribution combine function w.r.t. log_sigma2
// Inputs: mean - mu, log(variance) - log_sigma2, random numbers obey N(0,1) - epsilon
// Output: 0.5 * epsilon * exp(0.5*log_sigma2)
__device__ void NormRnd_drv_log_sigma2(float& output, const float& mu, const float& log_sigma2, const float& epsilon, const float& dummy1 = 0, const float& dummy2 = 0) {
    output = 0.5 * epsilon * exp( 0.5 * log_sigma2 );
}

// typedef these iterators for shorthand
typedef thrust::device_vector<float>::iterator FloatIterator;
// typedef a tuple of these iterators
typedef thrust::tuple<FloatIterator, FloatIterator, FloatIterator, FloatIterator, FloatIterator, FloatIterator> IteratorTuple;
// typedef the zip_iterator of this tuple
typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
// Struct to store function pointer of combine function and its partial derivatives
struct cmbFuncInfo {
    cmb_functor func;
    vector<cmb_functor> drv;
    thrust::device_vector<float>* inputPtr[CMB_MAX_ARG_NUM];
    ZipIterator iter_begin;
    ZipIterator iter_end;

    cmbFuncInfo (cmb_functor _func = cmb_functor(NULL), vector<cmb_functor> _drv = vector<cmb_functor>(1,cmb_functor(NULL)) ) : func(_func), drv(_drv) {}
    cmbFuncInfo& operator=(const cmbFuncInfo& other) {
        if (this != &other) {
            func = other.func;
            drv = other.drv;
            for (int i = 0; i < CMB_MAX_ARG_NUM; i++)
                inputPtr[i] = other.inputPtr[i];
        }
        return *this;
    }
    void setZipIterator(thrust::device_vector<float>& output) {
        iter_begin = thrust::make_zip_iterator(thrust::make_tuple(output.begin(), inputPtr[0]->begin(),
            inputPtr[1]->begin(), inputPtr[2]->begin(), inputPtr[3]->begin(), inputPtr[4]->begin()));
        iter_end = thrust::make_zip_iterator(thrust::make_tuple(output.end(), inputPtr[0]->end(), 
            inputPtr[1]->end(), inputPtr[2]->end(), inputPtr[3]->end(), inputPtr[4]->end()));
    }
};

// Get device function pointer through a kernel, the order should be same as the map func_ptr_idx
__global__ void get_address_of_device_cmb_function(cmb_func_type *ptr_to_ptr) 
{ 
    ptr_to_ptr[0] = &NormRnd;
    ptr_to_ptr[1] = &NormRnd_drv_mu;
    ptr_to_ptr[2] = &NormRnd_drv_log_sigma2;
} 

class cmbFunctionMap {
public:
    map<string, cmbFuncInfo> funcList;
    map<cmb_functor, string> funcNameList;
public:
    // Constructor: add all existing function to the map
    cmbFunctionMap()
    {
        map<string, int> func_ptr_idx;
        func_ptr_idx["NormRnd"] = 0;
        func_ptr_idx["NormRnd_drv_mu"] = 1;
        func_ptr_idx["NormRnd_drv_log_sigma2"] = 2;

        thrust::device_vector<cmb_func_type> func_ptr(func_ptr_idx.size()); 
        get_address_of_device_cmb_function<<<1,1>>>(raw_pointer_cast(func_ptr.data())); 

        // Use name to find functions
        funcList["null"] = cmbFuncInfo(cmb_functor(NULL), {cmb_functor(NULL)});
        funcList["NormRnd"] = cmbFuncInfo(cmb_functor(func_ptr[func_ptr_idx["NormRnd"]]),
            {cmb_functor(func_ptr[func_ptr_idx["NormRnd_drv_mu"]]), 
            cmb_functor(func_ptr[func_ptr_idx["NormRnd_drv_log_sigma2"]])});
        // Use function point to find its name
        funcNameList[cmb_functor(NULL)] = "null";
        funcNameList[cmb_functor(func_ptr[func_ptr_idx["NormRnd"]])] = "NormRnd";
    }
    
    // Destructor
    ~cmbFunctionMap()
    {
        funcList.clear();
        funcNameList.clear();
    }
    
    // [] overload: given name return the function and derivative
    cmbFuncInfo operator[](string name)
    {
        return funcList[name];
    }
    
    // () overload: given function return its name
    string operator()(cmb_functor func)
    {
        return funcNameList[func];
    }
};

#endif
