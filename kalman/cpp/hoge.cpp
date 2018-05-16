#include <iostream>
#include <queue>
#include "Eigen/Core"
using namespace std;

using namespace Eigen;

static const unsigned int dim = 3;
using PxyType = Matrix<double, dim, 1>;

template <int SIZE>
class Memory
{
public:
    Memory()
    {
        for (int i = 0; i < SIZE; i++) {
            m_pxy.log.push(PxyType::Zero());
        }
        m_pxy.sum = PxyType::Zero();
    }

    PxyType averagePxy() { return m_pxy.sum / SIZE; }

    void add(PxyType& pxy)
    {
        registerData(pxy, m_pxy);
    }

private:
    template <typename T>
    struct DataSet {
        std::queue<T> log;
        T sum;
    };

    template <typename Tdata>
    void registerData(Tdata& data, DataSet<Tdata>& pair)
    {
        pair.sum -= pair.log.front();
        pair.log.pop();
        pair.log.push(data);
        pair.sum += data;
    }

    DataSet<PxyType> m_pxy;
};

int main()
{
    //initialize

    //use
    static const long ave_range = 2;
    static Memory<ave_range> pack;  //pxytype専用 3x1行列

    for (int i = 0; i < 30; i++) {
        PxyType hoge;
        hoge << i, i * 2, i + 1;
        pack.add(hoge);
        cout << pack.averagePxy().transpose() << "\n";
    }
    return 0;
}