//
//  usefulFunctions.hpp
//  CreateTask
//
//  Created by Charles Findling on 22/09/2015.
//  Copyright © 2015 Charles Findling. All rights reserved.
//

#ifndef usefulFunctions_hpp
#define usefulFunctions_hpp

#include <stdio.h>
#include <vector>
#include <boost/random/mersenne_twister.hpp>
#include <iostream>

int Sample_Discrete_Distribution(boost::mt19937 &generator, double* probabilities, int dimension);

void stratified_resampling(double uniformSample, double weights[], int dim, int* ancestorsIndexes, double sum_weights);

std::vector<int> Sample_Discrete_Distribution(boost::mt19937 &generator, int min, int max, int numberOfSamples = 1);

std::vector<int> Sample_Discrete_Distribution(boost::mt19937 &generator, const std::vector<double>& probabilities, int numberOfSamples);

int Sample_Discrete_Distribution(boost::mt19937 &generator, const std::vector<double>& probabilities);

std::vector<double> Sample_Uniform_Distribution(boost::mt19937 &generator, int numberOfSamples);

double Sample_Uniform_Distribution(boost::mt19937 &generator);

std::vector<double> Sample_Beta_Distribution(boost::mt19937 &generator, double a, double b, int numberOfSamples = 1);

std::vector<double> Sample_Normal_Distribution(boost::mt19937 &generator, double mu, double std, int numberOfSamples);

double Sample_Normal_Distribution(boost::mt19937 &generator, double mu = 0., double std = 1.);

std::vector<double> Sample_Gamma_Distribution(boost::mt19937 &generator, double k= 1., double theta = 1., int numberOfSamples = 1);

std::vector<double> Sample_Dirichlet_Distribution(boost::mt19937 &generator, double* dirichletParam, int dim);

double log_beta_pdf(double x, double a, double b);

double log_dirichlet_pdf(double* sample, double* dirichletParam, int K);

std::vector<bool> adapted_logical_xor(std::vector<bool> const& states, bool const& rew);

std::vector<int> stratified_resampling(boost::mt19937 &generator, std::vector<double> &weights);

std::vector<int> stratified_resampling(boost::mt19937 &generator, double weights[], int dim);


template<typename T>
std::vector<double> isNotEqual_timesVector_normalise(T* const &a, int const& x, int const& xdim, int const& ydim, T const& b, double* gamma_pointer)
{
    assert(x < xdim);
    std::vector<double> res(ydim);
    double sum_vector = 0;
    for (int i = 0; i < ydim; ++i) {
        res[i] = (a[ydim * x + i] != b) * (*(gamma_pointer + i));
        sum_vector += res[i];
    }
    for (int i = 0; i < ydim; ++i) {
        res[i] = res[i] / sum_vector;
    }
    return res;
}


template<typename T>
std::vector<bool> isEqual_and_adapted_logical_xor(T* const &a, int const& x, int const& xdim, int const& ydim, T const& b, bool const& rew)
{
    assert(x < xdim);
    std::vector<bool> res(ydim);
    for (int i = 0; i < ydim; ++i) {
        res[i] = (!(a[ydim * x + i] == b) != rew);
    }
    return res;
}


template<typename T>
T sum(std::vector<T> &vect)
{
    T res = 0;
    typename std::vector<T>::iterator it;
    for (it = vect.begin(); it != vect.end(); ++it){
        res = res + (*it);
    }
    return res;
}

template<typename T>
T sum(T* array, int dim)
{
    T res = 0;
    for (int i = 0; i < dim; ++i){
        res += array[i];
    }
    return res;
}


template<typename T>
T prod(std::vector<T> &vect)
{
    T res = 1;
    typename std::vector<T>::iterator it;
    for (it = vect.begin(); it != vect.end(); ++it) {
        res = res * (*it);
    }
    return res;
}

template<typename T>
std::vector<bool> operator<=(std::vector<T> const& a, std::vector<T> const& b)
{
    assert(a.size() == b.size());
    unsigned long dim = a.size();
    std::vector<bool> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i] <= b[i];
    }
    return res;
}

template<typename T>
std::vector<bool> operator==(std::vector<T> const& a, std::vector<T> const& b)
{
    assert(a.size() == b.size());
    unsigned long dim = a.size();
    std::vector<bool> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i] == b[i];
    }
    return res;
}

template<typename T>
std::vector<bool> operator<=(std::vector<T> const &a, T const& b)
{
    unsigned long dim = a.size();
    std::vector<bool> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i] <= b;
    }
    return res;
}

template<typename T>
void divide(T* array, T divider, int dim)
{
    for (int i = 0; i < dim; ++i) {
        array[i] = array[i]/divider;
    }
    return;
}

template<typename T>
std::vector<bool> operator==(std::vector<T> const &a, T const& b)
{
    unsigned long dim = a.size();
    std::vector<bool> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i] == b;
    }
    return res;
}

template<typename T>
std::vector<bool> isEqual(T* const &a, int const& x, int const& xdim, int const& ydim, T const& b)
{
    assert(x < xdim);
    std::vector<bool> res(ydim);
    for (int i = 0; i < ydim; ++i) {
        res[i] = a[ydim * x + i] == b;
    }
    return res;
}

template<typename T>
std::vector<bool> operator!=(std::vector<T> const &a, T const& b)
{
    unsigned long dim = a.size();
    std::vector<bool> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = (a[i] != b);
    }
    return res;
}

template<typename T>
std::vector<bool> isNotEqual(T* const &a, int const& x, int const& xdim, int const& ydim, T const& b)
{
    assert(x < xdim);
    std::vector<bool> res(ydim);
    for (int i = 0; i < ydim; ++i) {
        res[i] = a[ydim * x + i] != b;
    }
    return res;
}
template<typename T>
std::vector<T> operator*(std::vector<bool> const &a, std::vector<T> const& b)
{
    assert(a.size() == b.size());
    unsigned long dim = a.size();
    std::vector<T> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i]*b[i];
    }
    return res;
}

template<typename T>
std::vector<T> operator*(std::vector<bool> const &a, T* const& b)
{
    unsigned long dim = a.size();
    std::vector<T> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i]*b[i];
    }
    return res;
}

template<typename T>
std::vector<T> operator*(std::vector<T> const &a, std::vector<T> const& b)
{
    assert(a.size() == b.size());
    unsigned long dim = a.size();
    std::vector<T> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i]*b[i];
    }
    return res;
}

template<typename T>
std::vector<T> operator*(T* const &a, std::vector<T> const& b)
{
    unsigned long dim = b.size();
    std::vector<T> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i]*b[i];
    }
    return res;
}

template<typename T>
std::vector<T> operator*(std::vector<T> const &a, T const& b)
{
    unsigned long dim = a.size();
    std::vector<T> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i]*b;
    }
    return res;
}

template<typename T>
std::vector<T> operator/(std::vector<T> const &a, T const& b)
{
    unsigned long dim = a.size();
    std::vector<T> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i]/b;
    }
    return res;
}

template<typename T>
std::vector<T> createRange(T const first, T const last, int const increment)
{
    int dim = (int)abs((last - first)/increment);
    std::vector<T> range(dim);
    for (int i = 0; i < dim; i++) {
        range[i] = first + i * increment;
    }
    return range;
}

template<typename T>
void print(std::vector<T> &vect){
    typename std::vector<T>::iterator it;
    for (it = vect.begin(); it != vect.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << " " << std::endl;
}

template<typename T>
double isNotEqual_timesVector_notnormalise(T* const &a, int const& x, int const& xdim, int const& ydim, T const& b, double* gamma_pointer, double* output)
{
    assert(x < xdim);
    double sum_vector = 0;
    for (int i = 0; i < ydim; ++i) {
        *(output + i) = (a[ydim * x + i] != b) * (*(gamma_pointer + i));
        sum_vector += *(output + i);
    }
/*    for (int i = 0; i < ydim; ++i) {
        *(output + i) = *(output + i) / sum_vector;
    }*/
    return sum_vector;
}


//std::vector<double> Sample_Weighted_Vector(boost::minstd_rand &generator, std::discrete_distribution<int> distribution, int numberOfSamples);
//int Sample_Weighted_Vector(std::default_random_engine &generator, std::discrete_distribution<int> &distribution);


#endif /* usefulFunctions_hpp */
