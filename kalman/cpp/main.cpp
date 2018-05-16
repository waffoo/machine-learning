#include <iostream>
#include "kalman.hpp"
using namespace std;

double uni_rand(double begin, double end)
{
    static std::random_device seed_gen;
    static std::default_random_engine engine(seed_gen());
    static std::uniform_real_distribution<> dist(begin, end);
    return dist(engine);
}

int main()
{
    int time = 100;
    for (int i = 0; i < time; i++) {
        double x1 = uni_rand(-5, 5), x2 = uni_rand(-5, 5);
        Craft::estimateParams(std::make_pair(x1, x2));
    }
}