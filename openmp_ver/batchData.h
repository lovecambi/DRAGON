#ifndef BATCHDATA_
#define BATCHDATA_

#include <iostream>
#include <vector>
#include <cmath>
#include <assert.h>
#ifdef __APPLE__
#include <libiomp/omp.h>
#else
#include <omp.h>
#endif

using namespace std;

template <typename T>
class batchData {
public:
	vector<T> data;
public:
    // Constructor
    batchData(size_t length = 1, T value = 0)	{   data = vector<T>(length, value);	}
    // Range constructor, mainly used for debugging
    batchData(int first, int last, int step)
    {
        assert(step != 0);
        assert((float)(last-first)/step > 0);
        int length = (last-first)/step + 1;
        data = vector<T>(length,0);
        for (int i = 0; i < length; i++)
            data[i] = first + i*step;
    }
	// Destructor
	~batchData()	{	vector<T>().swap(data);	}
    // Return size of batch data
    size_t size() const   {   return data.size(); }
    
    
    /* Indexing operator overloading */
    
    // [] overload: do indexing to get i-th element in data for reading
    T operator[](unsigned long index) const
    {
        assert(index >= 0 && index < data.size());
        return data[index];
    }
    // [] overload: do indexing to get i-th element in data for writing
    T& operator[](unsigned long index)
    {
        assert(index >= 0 && index < data.size());
        return data[index];
    }
    
    /* Assignment operator overloading */
    
	// = overload: copy assignment
	batchData<T>& operator=(const batchData<T>& other)
	{	
	    if (this != &other)
	    	data = other.data;
	    return *this;
	}
    
    // += overload: copy assignment
    batchData<T>& operator+=(const batchData<T>& other)
    {
        assert(size() == other.size());
        size_t batchSize = size();
#pragma simd
        for (int i = 0; i < batchSize; i++)
            data[i] += other.data[i];
        return *this;
    }
    
    
    /* Uniary operator overloading */
    
    // uniary - overload: opposite the vector
    friend batchData<T> operator-(const batchData<T>& v)
    {
        return v * (-1);
    }
    
    
    /* Pointwise binary operator overloading */
    
	// + overload: point-wise add of two vectors of same length
	friend batchData<T> operator+(const batchData<T>& v1, const batchData<T>& v2)
	{
        assert(v1.size() == v2.size());
		size_t batchSize = v1.size();
		batchData<T> ans(batchSize);
#pragma simd
		for (int i = 0; i < batchSize; i++)
			ans.data[i] = v1.data[i] + v2.data[i];
		return ans;
	}
    
    // - overload: point-wise substraction of two vectors of same length
    friend batchData<T> operator-(const batchData<T>& v1, const batchData<T>& v2)
    {
        return v1 + (-v2);
    }
    
    // * overload: point-wise multiplication of two vectors of same length
    friend batchData<T> operator*(const batchData<T>& v1, const batchData<T>& v2)
    {
        assert(v1.size() == v2.size());
        size_t batchSize = v1.size();
        batchData<T> ans(batchSize);
#pragma simd
        for (int i = 0; i < batchSize; i++)
            ans.data[i] = v1.data[i] * v2.data[i];
        return ans;
    }
    
    // / overload: point-wise divide of two vectors of same length
    friend batchData<T> operator/(const batchData<T>& v1, const batchData<T>& v2)
    {
        assert(v1.size() == v2.size());
        size_t batchSize = v1.size();
        batchData<T> ans(batchSize);
#pragma simd
        for (int i = 0; i < batchSize; i++)
            ans.data[i] = v1.data[i] / v2.data[i];
        return ans;
    }
    
	
    /* Vector-Scalor binary operator overloading */
    
	// + overload: right scalar addition of a vector
	friend batchData<T> operator+(const batchData<T>& v, const T& c)
	{
		size_t batchSize = v.size();
		batchData<T> ans(batchSize);
#pragma simd
		for (int i = 0; i < batchSize; i++)
			ans.data[i] = v.data[i] + c;
		return ans;
	}
    
    // + overload: left scalar addition of a vector
    friend batchData<T> operator+(const T& c, const batchData<T>& v)
    {
        return v + c;
    }
    
    // - overload: right scalar substraction of a vector
    friend batchData<T> operator-(const batchData<T>& v, const T& c)
    {
        return v + (-c);
    }
    
    // - overload: left scalar substraction of a vector
    friend batchData<T> operator-(const T& c, const batchData<T>& v)
    {
        return (-v) + c;
    }
    
	// * overload: right scalar multiplication of a vector
	friend batchData<T> operator*(const batchData<T>& v, const T& c)
	{
		size_t batchSize = v.size();
		batchData<T> ans(batchSize);
#pragma simd
		for (int i = 0; i < batchSize; i++)
			ans.data[i] = v.data[i] * c;
		return ans;
	}
    
    // * overload: left scalar multiplication of a vector
    friend batchData<T> operator*(const T& c, const batchData<T>& v)
    {
        return v * c;
    }
	
    // / overload: right scalar divide of a vector
    friend batchData<T> operator/(const batchData<T>& v, const T& c)
    {
        return v * ((T)1/c);
    }
    
    // / overload: left scalar divide of a vector
    friend batchData<T> operator/(const T& c, const batchData<T>& v)
    {
        size_t batchSize = v.size();
        batchData<T> ans(batchSize);
#pragma simd
        for (int i = 0; i < batchSize; i++)
            ans.data[i] = c/v.data[i];
        return ans;
    }
    
    
    /* Other operations */
    
	// Get summation of the vector's all elements
	T sum() {
		T ans(0);
		for (int i = 0; i < data.size(); i++)
			ans += data[i];
		return ans;
	}
    
    // Get product of the vector's all elements
    T prod() {
        T ans(1);
        for (int i = 0; i < data.size(); i++)
            ans *= data[i];
        return ans;
    }
    
    // Get 2-norm of the vector
    T norm() {
        T ans(0);
        for (int i = 0; i < data.size(); i++)
            ans += data[i] * data[i];
        return sqrt(ans);
    }
    
    // Normalize
    void normalize() {
        T s = sum();
        assert(s != 0);
#pragma simd
        for (int i = 0; i < size(); i++)
            data[i] /= s;
    }
    
    // Dot product of two vectors
    friend vector<T> dot(const batchData<T>& v1, const batchData<T>& v2, bool normalize)
    {
        assert(v1.size() == v2.size());
        size_t batchSize = v1.size();
        T ans(0);
        for (int i = 0; i < batchSize; i++)
            ans[i] += v1[i] * v2[i];
        if (normalize)
#pragma simd
            for (int i = 0; i < batchSize; i++)
                ans.data[i] /= batchSize;
        return ans;
    }
    
	// Dot product of one batchData and one vector of batchData's dot((1*B) , (N*B))
	friend vector<T> dot(const batchData<T>& v, const vector<batchData<T>>& vs, bool normalize = false)
	{
        for (int i = 0; i < vs.size(); i++)
            assert(v.size() == vs[i].size());
		size_t batchSize = v.size();
		size_t len = vs.size();
		vector<T> ans(len,0);
#pragma omp parallel for
		for (int i = 0; i < len; i++)
			for (int j = 0; j < batchSize; j++)
                ans[i] += normalize ? v[j] * vs[i][j] / batchSize : v[j] * vs[i][j];
		return ans;
	}

	// << overload: print the vector for debug
	friend ostream& operator<<(ostream& out, const batchData<T> &v)
	{	
		for (int i = 0; i < v.size(); i++) {
			out << v[i] << " ";
		}
		out << endl;
		return out;
	}

};

#endif