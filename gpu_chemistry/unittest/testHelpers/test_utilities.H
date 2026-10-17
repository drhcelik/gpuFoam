#pragma once

#include <cmath>
#include <cstdlib> //rand
#include <vector>


static inline double random_number(double LO, double HI){
    double r = LO + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(HI-LO)));
    return r;
}

template<class T>
static inline void fill_random(T& v, double LO = 0.0, double HI = 1.0)
{


    for (auto& e : v)
    {
        e = random_number(LO, HI);
    }

}

template<class T>
static inline void fill_linear(T& v){

    double dx = 1.0 / v.size();

    for (int i = 0; i < static_cast<int>(v.size()); ++i){
        v[i] = dx*i + dx;
    }



}

template<class C>
static inline void remove_negative(C& c, double tolerance){

    for (auto& e : c){
        if (e < tolerance){
            e = double(0);
        }
    }

}


// gpuFoam's GPU ODE state is [Y_0..Y_{n-1}, T, p] (nSpecie + 2 entries) while
// the OpenFOAM-dev reference (chemistryModels::Standard) integrates
// [Y_0..Y_{n-1}, T] (nSpecie + 1 entries) with p held constant.  These
// helpers drop the trailing pressure entry from a GPU-layout vector, or the
// trailing pressure row and column from a GPU-layout row-major square
// matrix, so that like entries are compared.

template<class T>
static inline std::vector<T> drop_pressure(const std::vector<T>& v)
{
    return std::vector<T>(v.begin(), v.end() - 1);
}

template<class T>
static inline std::vector<T> drop_pressure_matrix(const std::vector<T>& m)
{
    const int n = static_cast<int>(std::lround(std::sqrt(double(m.size()))));
    std::vector<T> ret;
    ret.reserve((n - 1)*(n - 1));
    for (int i = 0; i < n - 1; ++i)
    {
        for (int j = 0; j < n - 1; ++j)
        {
            ret.push_back(m[i*n + j]);
        }
    }
    return ret;
}
