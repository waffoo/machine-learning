#include "kalman.hpp"
#include "Eigen/Core"
#include "Eigen/LU"
#include "Eigen/Cholesky"
#include <iostream>

namespace Craft
{
using namespace Eigen;

static const unsigned int dim = 3;
using P_vec = Matrix<double, dim, 1>;  //P is Parameter
using P_square = Matrix<double, dim, dim>;
using AnsVec = Matrix<double, 1, 1>;
using AnsSquare = Matrix<double, 1, 1>;
using PxyType = Matrix<double, dim, 1>;
using Input = std::pair<double, double>;

//答え
constexpr int coef1 = 2, coef2 = 7, coef3 = 1;

void ukfInitialize(P_vec& params)
{
    params(0, 0) = 0;
    params(1, 0) = 0;
    params(2, 0) = 0;
}

double normal_rand(double dis)  //正規分布に従う乱数を返す関数
{
    static std::random_device seed_gen;
    static std::default_random_engine engine(seed_gen());
    static std::normal_distribution<> dist(0, dis);
    return dist(engine);
}


void estimateParams(Input input)  //2入力
{
    //下準備
    static const double alpha = 1e-3f + 1;
    static const double beta = 2.0f;
    static const double lambda = (alpha * alpha - 1) * dim;
    static const double weight_s0 = lambda / (dim + lambda);
    static const double weight_c0 = weight_s0 + 1 - alpha * alpha + beta;
    static const double weight_others = 0.5f / (dim + lambda);


    static P_vec params;
    static P_square P;  //paramsの分散共分散行列
    static const double noise_size = 1.0f;
    static const AnsSquare obs_noise
        = AnsSquare::Identity() * noise_size;  //観測ノイズの分散
    static const double stim_size = 0;         //刺激の大きさ
    static const long stim_interval = 100;     //刺激の間隔
    static const P_square stim
        = P_square::Identity() * stim_size;  //刺激
    static bool first = true;


    if (first) {  //params, Pの初期化
        ukfInitialize(params);
        for (unsigned int i = 0; i < dim; i++) {
            if (params(i, 0) != 0) {
                P(i, i) = params(i, 0) * 0.1;
            } else {
                P(i, i) = 1;
            }
        }
        first = false;
    }
    static long cnt = 0;
    cnt++;
    if (cnt % stim_interval == 0)
        P += stim;

    LLT<P_square> calc(P * (dim + lambda));
    P_square L = calc.matrixL();

    //sample_paramsの錬成
    std::array<P_vec, dim * 2 + 1> sample_params;
    sample_params[0] = params;
    for (unsigned int i = 0; i < dim; i++) {
        sample_params[i * 2 + 1] = params + L.col(i);
        sample_params[i * 2 + 2] = params - L.col(i);
    }

    //sample_paramsを変換
    std::array<AnsVec, dim * 2 + 1> sample_results;
    for (unsigned int i = 0; i < sample_params.size(); i++) {
        sample_results[i](0, 0) = sample_params[i](0, 0) * input.first + sample_params[i](1, 0) * input.second + sample_params[i](2, 0);
    }

    //変換結果の平均
    AnsVec ave_result = AnsVec::Zero();
    for (unsigned int i = 0; i < sample_results.size(); i++) {
        double weight = (i == 0 ? weight_s0 : weight_others);
        ave_result += weight * sample_results[i];
    }

    //共分散行列の計算
    PxyType P_xy = PxyType::Zero();
    AnsSquare P_yy = obs_noise;
    P_vec diff_x;
    AnsVec diff_y;
    for (unsigned int i = 0; i < sample_params.size(); i++) {
        double weight = (i == 0 ? weight_c0 : weight_others);
        diff_x = sample_params[i] - params;
        diff_y = sample_results[i] - ave_result;
        P_xy += weight * diff_x * diff_y.transpose();
        P_yy += weight * diff_y * diff_y.transpose();
    }

    //観測値をとってくる
    AnsVec obs_ans;
    obs_ans << coef1 * input.first + coef2 * input.second + coef3 + normal_rand(1.0);

    //更新
    PxyType gain = P_xy * P_yy.inverse();
    params += gain * (obs_ans - ave_result);
    P -= gain * P_xy.transpose();

    std::cout << params.transpose() << std::endl;
}
}