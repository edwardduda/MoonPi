#include <pybind11/pybind11.h>
#include <cmath>
#include <pybind11/numpy.h>
namespace py = pybind11;

float normalize_reward(float reward) {
    if(std::abs(reward) > 1.0){
        if(reward < 0){
            return -std::log(1 + std::abs(reward));
        }
        else{
            return std::log(1 + std::abs(reward));
        }
    } 
    return 0.0;
}

PYBIND11_MODULE(MarketEnvCpp, m) {
    m.doc() = "Fast C++ implementations for MarketEnv";
    m.def("normalize_reward", &normalize_reward, "Calculate normalized trading reward");
}