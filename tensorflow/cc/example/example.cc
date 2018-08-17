//
// Created by Zeng,Jinle on 2018/8/16.
//

#include <iostream>
#include <string>
#include <random>
#include <memory>
#include <algorithm>
#include <numeric>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <gperftools/profiler.h>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

using namespace tensorflow;
using namespace tensorflow::ops;

template <typename T>
inline const T *Get(const Tensor &tensor) {
    return reinterpret_cast<const T*>(tensor.tensor_data().data());
}

template <typename T>
inline T *GetMutable(Tensor &tensor) {
    return const_cast<T*>(Get<T>(tensor));
}

inline std::unique_ptr<FusedBatchNorm> BatchNormInferPrepare(const Scope &scope, const Tensor &x, const Tensor &scale, const Tensor &bias,
    const Tensor &mean, const Tensor &var, float epsilon, const std::string &data_layout) {
    return std::unique_ptr<FusedBatchNorm>(new FusedBatchNorm(scope, x, scale, bias, mean, var,
                          FusedBatchNorm::Attrs().Epsilon(epsilon).DataFormat(data_layout.c_str()).IsTraining(false)));
}

struct ThreadContext {
    ThreadContext(ClientSession &session, FusedBatchNorm &batch_norm)
            : session_(session), batch_norm_(batch_norm) {}

    ClientSession &session_;
    FusedBatchNorm &batch_norm_;
    std::vector<Tensor> outputs_;

    void Run() {
        session_.Run({batch_norm_.y}, &outputs_);
    }
};

void TestMain(const std::string &data_layout, int channel_size, int iters)  {
    //std::cerr << std::endl;

    const int64 n = 1, h = 224, w = 224;
    //const int64 channel_size = 3;
    const float epislon = 1e-6;
    //const std::string data_layout = "NCHW";
    using DataType = float;
    constexpr auto kTensorDataType = DT_FLOAT;

    int cpu_count = std::thread::hardware_concurrency();
    //std::cerr << "Hardware concurrency: " << cpu_count << std::endl;
    cpu_count = std::min(cpu_count, 10);

    std::vector<std::unique_ptr<Scope>> scope(cpu_count);
    std::vector<std::unique_ptr<ClientSession>> session(cpu_count);
    std::vector<std::unique_ptr<ThreadContext>> thread_context(cpu_count);

    std::vector<std::unique_ptr<Tensor>> x(cpu_count);
    std::vector<std::unique_ptr<Tensor>> scale(cpu_count);
    std::vector<std::unique_ptr<Tensor>> bias(cpu_count);
    std::vector<std::unique_ptr<Tensor>> mean(cpu_count);
    std::vector<std::unique_ptr<Tensor>> var(cpu_count);
    std::vector<std::unique_ptr<FusedBatchNorm>> op(cpu_count);

    for (int i = 0; i < cpu_count; ++i) {
        scope[i].reset(new Scope(Scope::NewRootScope()));
        Scope& root = *scope[i];
        session[i].reset(new ClientSession(root));

        if (data_layout == "NCHW") {
            x[i].reset(new Tensor(kTensorDataType, {n, channel_size, h, w}));
        } else {
            x[i].reset(new Tensor(kTensorDataType, {n, h, w, channel_size}));
        }

        scale[i].reset(new Tensor(kTensorDataType, {channel_size}));
        bias[i].reset(new Tensor(kTensorDataType, {channel_size}));
        mean[i].reset(new Tensor(kTensorDataType, {channel_size}));
        var[i].reset(new Tensor(kTensorDataType, {channel_size}));
        auto *var_data = GetMutable<DataType>(*var[i]);
        //std::cerr << var[i]->NumElements() << " " << var_data << std::endl;
        for (int j = 0; j < var[i]->NumElements(); ++j) {
          var_data[j] = j + 1;
        }

        op[i] = BatchNormInferPrepare(root.WithOpName("y"), *x[i], *scale[i], *bias[i], *mean[i], *var[i], epislon, data_layout);

        thread_context[i].reset(new ThreadContext(*session[i], *op[i]));
    }

    std::vector<std::unique_ptr<std::thread>> threads(cpu_count);
    std::vector<double> time_cost_ms(cpu_count);
    for (int cur_cpu_count = 0; cur_cpu_count < cpu_count; ++cur_cpu_count) {
        std::condition_variable start_cv;
        std::mutex mutex;
        bool start_flag = false;

        for (int i = 0; i <= cur_cpu_count; ++i) {
            threads[i].reset(new std::thread([&, i]() {
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    start_cv.wait(lock, [&] { return start_flag; });
                }
                for (int j = 0; j < iters; ++j) {
                    thread_context[i]->Run();
                }
            }));
        }

        std::string profile_output = "/Paddle/tensorflow/prof_thread_" + std::to_string(cur_cpu_count) + "_layout_" + data_layout + "_channel_" + std::to_string(channel_size) + ".prof";
        ProfilerStart(profile_output.c_str());
        //auto start_time = std::chrono::high_resolution_clock::now();
        {
            std::unique_lock<std::mutex> lock(mutex);
            start_flag = true;
            start_cv.notify_all();
        }
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i <= cur_cpu_count; ++i) {
            threads[i]->join();
        }
        ProfilerStop();
        auto time_cost = std::chrono::high_resolution_clock::now() - start_time;
        time_cost_ms[cur_cpu_count] =
                std::chrono::duration_cast<std::chrono::nanoseconds>(time_cost).count() / static_cast<double>(1e6);
    }

    std::cerr << std::endl;
    for (int i = 0; i < cpu_count; ++i) {
        double speed_up = time_cost_ms[0] / time_cost_ms[i] * (i+1);
        double slope = speed_up/(i+1);
        std::cerr << "Data Layout: " << data_layout << ", channel_size: " << channel_size << ", iters:" << iters << std::endl;
        std::cerr << "CPU number: " << i+1 << ", time:" << time_cost_ms[i] << "ms, ";
        std::cerr << "speed_up: " << speed_up << ", slope: " << slope << ", "; 
        std::cerr << "sum_of_ret: ";
        auto& ret = thread_context[i]->outputs_[0];
        std::cerr << std::accumulate(Get<DataType>(ret), Get<DataType>(ret) + ret.NumElements(), static_cast<DataType>(0)) << std::endl;
    }
}

int main()
{
  std::vector<int> channel_sizes({3, 10, 20, 30});
  std::vector<std::string> data_layouts({"NCHW", "NHWC"});
  for (auto &channel_size : channel_sizes) {
    for (auto &data_layout : data_layouts) {
      TestMain(data_layout, channel_size, 1<<12);
    }
  }
}

