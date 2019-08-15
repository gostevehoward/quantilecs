#ifndef QUANTILESIM_QUANTILE_BAI_H_
#define QUANTILESIM_QUANTILE_BAI_H_

#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "uniform_boundaries.h"

using ArmGenerator = std::function<double(double, std::mt19937_64&)>;

class BanditModel {
 public:
  static BanditModel NITHBernoulli(const int num_arms, const double quantile_p,
                                   const double p_prime) {
    assert(quantile_p > p_prime / 2);
    assert(1 - quantile_p > p_prime / 2);
    auto means = std::vector<double>(num_arms, 1 - quantile_p - p_prime / 2);
    means[0] += p_prime;
    ArmGenerator generator = [](double mean, std::mt19937_64& rng) {
      auto dist = std::bernoulli_distribution(mean);
      return dist(rng);
    };
    return BanditModel(means, generator);
  }

  static BanditModel NITHUniform(const int num_arms, const double shift) {
    auto minima = std::vector<double>(num_arms, 0);
    minima[0] = shift;
    ArmGenerator generator = [](double minimum, std::mt19937_64& rng) {
      std::uniform_real_distribution<double> dist;
      return dist(rng) + minimum;
    };
    return BanditModel(minima, generator);
  }

  static BanditModel NITHCauchy(const int num_arms, const double shift) {
    auto locations = std::vector<double>(num_arms, 0);
    locations[0] = shift;
    ArmGenerator generator = [](double location, std::mt19937_64& rng) {
      std::cauchy_distribution<double> dist;
      return dist(rng) + location;
    };
    return BanditModel(locations, generator);
  }

  static BanditModel NormalSpread(const int num_arms) {
    auto std_devs = std::vector<double>(num_arms, 1);
    std_devs[0] = 2;
    ArmGenerator generator = [](double std_dev, std::mt19937_64& rng) {
      std::normal_distribution<double> dist(0, std_dev);
      return dist(rng);
    };
    return BanditModel(std_devs, generator);
  }

  BanditModel(std::vector<double> locations, ArmGenerator arm_generator)
      : locations_(locations), arm_generator_(arm_generator) {}

  BanditModel(const BanditModel& other) = default;

  BanditModel& operator=(BanditModel&& other) {
    locations_ = std::move(other.locations_);
    arm_generator_ = std::move(other.arm_generator_);
    return *this;
  }

  int num_arms() const {
    return locations_.size();
  }

  double sample_arm(const int arm_index, std::mt19937_64& rng) const;
  void dump() const;

 private:
  std::vector<double> locations_;
  ArmGenerator arm_generator_;
};

class ConfidenceInterval {
 public:
  virtual ~ConfidenceInterval() {}
  virtual double radius(const int num_samples, const double error_rate) const
  = 0;
};

class SzorenyiCI : public ConfidenceInterval {
 public:
  double radius(const int num_samples, const double error_rate) const override {
    return sqrt(1.0 / (2 * num_samples)
                * log(pow(M_PI * num_samples, 2) / (3 * error_rate)));
  }
};

class BetaBinomialCI : public ConfidenceInterval {
 public:
  BetaBinomialCI(const double p, const double t_opt, const double alpha_opt)
      :  p_(p), mixture_(p * (1 - p) * t_opt, alpha_opt, p, 1 - p, true) {}

  double radius(const int t, const double error_rate) const
      override {
    return mixture_.bound(p_ * (1 - p_) * t, log(1 / error_rate)) / t;
  }

 private:
  const double p_;
  const confseq::BetaBinomialMixture mixture_;
};

class StitchedCI : public ConfidenceInterval {
 public:
  StitchedCI(const double p, const int min_sample_size, const double eta,
             const double s)
      : p_(p), bound_(p * (1 - p) * min_sample_size, (1 - 2 * p) / 3, s, eta) {}

  double radius(const int t, const double error_rate) const override {
    return bound_(p_ * (1 - p_) * t, error_rate) / t;
  }

  const double p_;
  const confseq::PolyStitchingBound bound_;
};

class TreeNode {
  public:

  static TreeNode* const NULL_NODE;

  static void delete_subtree(TreeNode* node) {
    if (!node->is_null()) {
      delete_subtree(node->left_);
      delete_subtree(node->right_);
      delete node;
    }
  }

  TreeNode() : value_(0), subtree_count_(0), subtree_height_(0), left_(NULL),
               right_(NULL), parent_(NULL) {}
  TreeNode(double value)
      : value_(value), subtree_count_(1), subtree_height_(1),
        left_(NULL_NODE), right_(NULL_NODE), parent_(NULL) {}

  TreeNode* insert(const double value);
  double search(const int target_count, const int count_leq_above) const;
  double count_less(const double target_value, const int count_less_above)
      const;
  double count_less_or_equal(const double target_value,
                             const int count_leq_above) const;
  const TreeNode* next() const;
  const TreeNode* first_in_subtree() const;

  TreeNode* rotate_left();
  TreeNode* rotate_right();
  TreeNode* rotate_right_left();
  TreeNode* rotate_left_right();

  bool is_null() const {
    return subtree_count_ == 0;
  }

  int balance_factor() const {
    return right_->subtree_height_ - left_->subtree_height_;
  }

  void refresh() {
    assert(!is_null());
    subtree_count_ = left_->subtree_count_ + right_->subtree_count_ + 1;
    subtree_height_ =
        std::max(left_->subtree_height_, right_->subtree_height_) + 1;
  }

  double value_;
  int subtree_count_;
  int subtree_height_;
  TreeNode* left_;
  TreeNode* right_;
  TreeNode* parent_;
};

class OrderStatisticTracker : public confseq::OrderStatisticInterface {
  public:

  virtual ~OrderStatisticTracker() {}
  virtual void insert(const double value) = 0;
};

class TreeTracker : public OrderStatisticTracker {
  public:

  TreeTracker() : root_(TreeNode::NULL_NODE) {}

  ~TreeTracker() {
    TreeNode::delete_subtree(root_);
  }

  void insert(const double value) override;
  double get_order_statistic(const int order_index) const override;
  int count_less(const double value) const override;
  int count_less_or_equal(const double value) const override;
  int size() const override;

 private:
  TreeNode* root_;
};

template <class ValueType> class Cached {
 public:
  ValueType get(std::function<ValueType()> refresh_fn) const {
    if (is_dirty_) {
      value_ = refresh_fn();
      is_dirty_ = false;
    }
    return value_;
  }

  void invalidate() { is_dirty_ = true; }

 private:
  mutable ValueType value_;
  mutable bool is_dirty_ = true;
};

class QuantileCI {
 public:
  void invalidate() {
    lower_.invalidate();
    upper_.invalidate();
    point_estimate_.invalidate();
  }

  Cached<double> lower_;
  Cached<double> upper_;
  Cached<double> point_estimate_;
};

class QuantileCIInterface {
 public:
  virtual ~QuantileCIInterface() {}
  virtual void insert(const int arm_index, const double value) = 0;
  virtual double lower_bound(const int arm_index, const double error_rate,
                             const bool add_epsilon) const = 0;
  virtual double upper_bound(const int arm_index, const double error_rate,
                             const bool add_epsilon) const = 0;
  virtual double point_estimate(const int arm_index, const bool add_epsilon)
      const = 0;
  virtual int num_arms() const = 0;
};

using MakeOSTracker = std::function<std::unique_ptr<OrderStatisticTracker>()>;

class QuantileCITracker : public QuantileCIInterface {
 public:
  QuantileCITracker(
      const int num_arms, const double quantile_p, const double epsilon,
      MakeOSTracker make_os_tracker,
      std::unique_ptr<const ConfidenceInterval>&& p_lower_ci,
      std::unique_ptr<const ConfidenceInterval>&& p_upper_ci,
      std::unique_ptr<const ConfidenceInterval>&& p_epsilon_lower_ci,
      std::unique_ptr<const ConfidenceInterval>&& p_epsilon_upper_ci)
      : quantile_p_(quantile_p), epsilon_(epsilon),
        p_lower_ci_(std::move(p_lower_ci)), p_upper_ci_(std::move(p_upper_ci)),
        p_epsilon_lower_ci_(std::move(p_epsilon_lower_ci)),
        p_epsilon_upper_ci_(std::move(p_epsilon_upper_ci)),
        os_trackers_(num_arms), p_ci_cache_(num_arms),
        p_epsilon_ci_cache_(num_arms) {
    for (int arm = 0; arm < num_arms; arm++) {
      os_trackers_[arm] = make_os_tracker();
    }
  }

  void insert(const int arm_index, const double value) override;
  double lower_bound(const int arm_index, const double error_rate,
                     const bool add_epsilon) const override;
  double upper_bound(const int arm_index, const double error_rate,
                     const bool add_epsilon) const override;
  double point_estimate(const int arm_index, const bool add_epsilon) const
      override;
  int num_arms() const override {return os_trackers_.size();}

 private:
  double get_quantile(const int arm_index, const double p) const;

  const double quantile_p_;
  const double epsilon_;
  const std::unique_ptr<const ConfidenceInterval> p_lower_ci_;
  const std::unique_ptr<const ConfidenceInterval> p_upper_ci_;
  const std::unique_ptr<const ConfidenceInterval> p_epsilon_lower_ci_;
  const std::unique_ptr<const ConfidenceInterval> p_epsilon_upper_ci_;
  std::vector<std::unique_ptr<OrderStatisticTracker>> os_trackers_;
  std::vector<QuantileCI> p_ci_cache_;
  std::vector<QuantileCI> p_epsilon_ci_cache_;
};

class Agent {
 public:
  static const int NO_ARM_SELECTED;

  virtual ~Agent() {}
  virtual int get_arm_to_sample() const = 0;
  virtual int update(const int arm_pulled, const double value) = 0;
};

class QpacAgent : public Agent {
  public:

  QpacAgent(const double error_rate,
            std::unique_ptr<QuantileCIInterface>&& ci_tracker)
      : error_rate_(error_rate / ci_tracker->num_arms()),
        ci_tracker_(std::move(ci_tracker)) {
    for (int i = 0; i < ci_tracker_->num_arms(); i++) {
      active_arms_.insert(i);
    }
    next_arm_iter_ = active_arms_.begin();
  }

  int get_arm_to_sample() const override;
  int update(const int arm_pulled, const double value) override;

 private:
  const double error_rate_;
  std::unique_ptr<QuantileCIInterface> ci_tracker_;
  std::unordered_set<int> active_arms_;
  std::unordered_set<int>::iterator next_arm_iter_;
};

class DoubledMaxQAgent : public Agent {
  public:

  DoubledMaxQAgent(const int num_arms, const double error_rate,
                   const double epsilon, const double quantile_p,
                   MakeOSTracker make_os_tracker)
      : quantile_p_(quantile_p),
        L_D_(compute_L_D(num_arms, error_rate, epsilon, quantile_p)),
        N_0_(floor(3 * L_D_ / (1 - quantile_p)) + 1),
        stop_sample_size_(10 * (1 - quantile_p) * L_D_ / (epsilon * epsilon)),
        os_trackers_(num_arms) {
    samples_remaining_ = N_0_;
    for (int arm = 0; arm < num_arms; arm++) {
      os_trackers_[arm] = make_os_tracker();
    }
  }

  int get_arm_to_sample() const override;
  int update(const int arm_pulled, const double value) override;

 private:
  static inline double compute_L_D(int num_arms, double error_rate,
                                   double epsilon, double quantile_p) {
    return 6 * log(num_arms * log(20 * (1 - quantile_p) * log(1 / error_rate)
                                  / (epsilon * epsilon)) / log(2))
        - log(error_rate);
  }

  const double quantile_p_;
  const double L_D_;
  const double N_0_;
  const double stop_sample_size_;
  int next_arm_to_sample_ = 0;
  int samples_remaining_;
  bool is_initial_sampling_done_ = false;
  std::vector<std::unique_ptr<OrderStatisticTracker>> os_trackers_;
};


class LucbAgent : public Agent {
  public:

  LucbAgent(const double error_rate,
            std::unique_ptr<QuantileCIInterface>&& ci_tracker)
      : error_rate_(error_rate / ci_tracker->num_arms()),
        tracker_(std::move(ci_tracker)),
        initial_sampling_index_(tracker_->num_arms() - 1) {}

  int get_arm_to_sample() const override;
  int update(const int arm_pulled, const double value) override;

 private:
  int best_arm() const;
  int best_competitor() const;
  bool can_stop() const;

  const double error_rate_;
  std::unique_ptr<QuantileCIInterface> tracker_;
  bool sample_best_arm_next_ = false;
  int initial_sampling_index_;
};

class ABTestAgent : public Agent {
 public:
  ABTestAgent(const double error_rate, const double quantile_p, const int t_opt,
              MakeOSTracker make_os_tracker)
    : error_rate_(error_rate),
      quantile_p_(quantile_p),
      os_trackers_({make_os_tracker(), make_os_tracker(), make_os_tracker()}),
      ab_test_(quantile_p, t_opt, error_rate, os_trackers_[0], os_trackers_[1]),
      next_arm_(0) {}

  int get_arm_to_sample() const override;
  int update(const int arm_pulled, const double value) override;

 private:
  double best_arm() const;
  double empirical_quantile(const int arm) const;

  const double error_rate_;
  const double quantile_p_;
  std::vector<std::shared_ptr<OrderStatisticTracker>> os_trackers_;
  confseq::QuantileABTest ab_test_;
  int next_arm_;
  std::multiset<double> values[2];
};

struct RunResult {
  int chosen_arm;
  int num_rounds;
};

class SimulationConfig {
 public:
  SimulationConfig(int num_replications, int max_rounds, int num_threads,
                   double quantile_p, double epsilon)
      : num_replications_(num_replications), max_rounds_(max_rounds),
        num_threads_(std::min(num_replications, num_threads)),
        quantile_p_(quantile_p), epsilon_(epsilon),
        model_(BanditModel::NITHUniform(2, 1)) {
    assert(quantile_p + epsilon < 1);
    make_os_tracker_ = []() {
      return std::make_unique<TreeTracker>();
    };
  }

  SimulationConfig& error_rate(double rate) {
    error_rate_ = rate;
    return *this;
  }

  SimulationConfig& bernoulli_model(int num_arms, double shift) {
    model_ = BanditModel::NITHBernoulli(num_arms, quantile_p_, shift);
    return *this;
  }

  SimulationConfig& uniform_model(int num_arms, double shift) {
    model_ = BanditModel::NITHUniform(num_arms, shift);
    return *this;
  }

  SimulationConfig& cauchy_model(int num_arms, double shift) {
    model_ = BanditModel::NITHCauchy(num_arms, shift);
    return *this;
  }

  SimulationConfig& normal_spread_model(int num_arms) {
    model_ = BanditModel::NormalSpread(num_arms);
    return *this;
  }

  SimulationConfig& enable_ab_test() {
    ab_test_enabled_ = true;
    epsilon_ = 0;
    return *this;
  }

  int num_arms() const {
    return model_.num_arms();
  }

  const int num_replications_;
  const int max_rounds_;
  const int num_threads_;
  const double quantile_p_;
  double epsilon_;
  double error_rate_ = 0.05;
  BanditModel model_;
  MakeOSTracker make_os_tracker_;
  bool ab_test_enabled_ = false;
};

using AgentMap = std::unordered_map<std::string, std::shared_ptr<Agent>>;
using ResultMap = std::unordered_map<std::string, RunResult>;

void simulate_many_walks(const int thread_id,
                         const SimulationConfig* const config,
                         const bool* const stop_all_threads,
                         std::vector<ResultMap>* all_results);

#endif // QUANTILESIM_QUANTILE_BAI_H_
