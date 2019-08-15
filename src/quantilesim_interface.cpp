#include <chrono>
#include <future>
#include <thread>
#include <vector>

#include "quantile_bai.h"

#include <RcppCommon.h>

RCPP_EXPOSED_CLASS(SimulationConfig)

#include <Rcpp.h>

Rcpp::List run_simulation(const SimulationConfig& config) {
  const int actual_num_replications =
      (config.num_replications_ / config.num_threads_) * config.num_threads_;


  bool stop_all_threads = false;
  std::vector<ResultMap> all_results(actual_num_replications);
  std::vector<std::future<void>> futures;
  auto start = std::chrono::steady_clock::now();

  if (config.num_threads_ == 1) {
    //ProfilerStart("/Users/steve/temp/profiler.out");
    simulate_many_walks(0, &config, &stop_all_threads, &all_results);
    //ProfilerStop();
  } else {
    for (int thread_id = 0; thread_id < config.num_threads_; thread_id++) {
      futures.push_back(std::async(std::launch::async, simulate_many_walks,
                                   thread_id, &config, &stop_all_threads,
                                   &all_results));
    }

    int num_finished = 0;
    while (!stop_all_threads && num_finished < config.num_threads_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      try {
        Rcpp::checkUserInterrupt();
      } catch (Rcpp::internal::InterruptedException) {
        fprintf(stderr, "Caught interrupt, stopping threads\n");
        stop_all_threads = true;
        for (auto& future : futures) {
          future.wait();
        }
        throw;
      }
      num_finished = 0;
      for (auto& future : futures) {
        auto status = future.wait_for(std::chrono::milliseconds(0));
        if (status == std::future_status::ready) ++num_finished;
      }
    }
    for (auto& future : futures) future.get(); // check for exceptions
  }

  auto end = std::chrono::steady_clock::now();
  int elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
  fprintf(stderr, "Time elapsed=%.3fs\n", elapsed_ms / 1000.0);

  auto R_results = Rcpp::List::create();
  for (ResultMap result_map : all_results){
    auto run_list = Rcpp::List::create();
    for (auto name_result_pair : result_map) {
      RunResult result = name_result_pair.second;
      run_list(name_result_pair.first) = Rcpp::List::create(
          Rcpp::_["num_rounds"] = result.num_rounds,
          Rcpp::_["chosen_arm"] = result.chosen_arm);
    }
    R_results.push_back(run_list);
  }

  return R_results;
}

RCPP_MODULE(quantile_bai_cpp) {
  Rcpp::class_<SimulationConfig>("SimulationConfig")
      .constructor<int, int, int, double, double>()
      .method("error_rate", &SimulationConfig::error_rate)
      .method("bernoulli_model", &SimulationConfig::bernoulli_model)
      .method("uniform_model", &SimulationConfig::uniform_model)
      .method("cauchy_model", &SimulationConfig::cauchy_model)
      .method("normal_spread_model", &SimulationConfig::normal_spread_model)
      .method("enable_ab_test", &SimulationConfig::enable_ab_test);

  Rcpp::function("run_simulation", &run_simulation);
}
