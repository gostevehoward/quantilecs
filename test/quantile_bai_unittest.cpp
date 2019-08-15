#include <random>

#include "quantile_bai.h"

#include "gtest/gtest.h"

namespace {

TEST(BanditModelTest, Bernoulli) {
  std::mt19937_64 rng(123);
  auto model = BanditModel::NITHBernoulli(2, .5, .1);
  double draw = model.sample_arm(0, rng);
  EXPECT_TRUE(draw == 0.0 || draw == 1.0);
}

TEST(BanditModelTest, Uniform) {
  std::mt19937_64 rng(123);
  auto model = BanditModel::NITHUniform(2, 1);
  double draw1 = model.sample_arm(0, rng);
  EXPECT_LE(1, draw1);
  EXPECT_LE(draw1, 2);
  double draw2 = model.sample_arm(0, rng);
  EXPECT_NE(draw1, draw2);

  double draw3 = model.sample_arm(1, rng);
  EXPECT_LE(0, draw3);
  EXPECT_LE(draw3, 1);
}

TEST(BanditModelTest, Cauchy) {
  std::mt19937_64 rng(123);
  auto model = BanditModel::NITHCauchy(2, 2e6);
  double draw1 = model.sample_arm(0, rng);
  EXPECT_LE(1e6, draw1);
  double draw2 = model.sample_arm(0, rng);
  EXPECT_NE(draw1, draw2);
  double draw3 = model.sample_arm(1, rng);
  EXPECT_LE(draw3, 1e6);
}

TEST(SzorenyiCITest, Radius) {
  SzorenyiCI ci;
  EXPECT_NEAR(ci.radius(100, .05), 0.2588138, 1e-5);
}

TEST(BetaBinomialCITest, Radius) {
  // expected values based on R confseq package
  // beta_binomial_mixture_bound(100*.5*.5, .05, 10*.5*.5, .5, .5) / 100
  BetaBinomialCI ci(0.5, 10, .05);
  EXPECT_NEAR(ci.radius(100, .05), 0.1599309, 1e-5);
  // beta_binomial_mixture_bound(100*.9*.1, .05, 10*.9*.1, .9, .1) / 100
  BetaBinomialCI ci2(0.9, 10, .05);
  EXPECT_NEAR(ci2.radius(100, .05), 0.08109037, 1e-5);
}

TEST(StitchedCITest, Radius) {
  // expected values based on R confseq package
  // poly_stitching_bound(100*.5*.5, .05, 10*.5*.5, c=(1 - 2 * .5) / 3,
  //   eta=2.04) / 100
  StitchedCI ci(0.5, 10, 2.04, 1.4);
  EXPECT_NEAR(ci.radius(100, .05), 0.178119, 1e-5);
  // poly_stitching_bound(100*.9*.1, .05, 10*.9*.1, c=(1 - 2 * .9) / 3,
  //   eta=2.04) / 100
  StitchedCI ci2(0.9, 10, 2.04, 1.4);
  EXPECT_NEAR(ci2.radius(100, .05), 0.0888042, 1e-5);
}

TEST(TreeTrackerTest, BasicUse) {
  TreeTracker tracker;
  EXPECT_EQ(tracker.size(), 0);

  tracker.insert(1);
  EXPECT_EQ(tracker.size(), 1);

  tracker.insert(2);
  EXPECT_EQ(tracker.get_order_statistic(1), 1);
  EXPECT_EQ(tracker.get_order_statistic(2), 2);

  for (int i = 3; i <= 50; i++) {
    tracker.insert(i);
  }
  for (int i = 100; i >= 51; i--) {
    tracker.insert(i);
  }
  for (int i = 3; i <= 100; i++) {
    EXPECT_EQ(tracker.get_order_statistic(i), i);
    EXPECT_EQ(tracker.count_less_or_equal(i), i);
    EXPECT_EQ(tracker.count_less(i), i - 1);
  }
}

TEST(TreeTrackerTest, TieHandling) {
  TreeTracker tracker;
  tracker.insert(0);
  tracker.insert(1);
  tracker.insert(0);
  EXPECT_EQ(tracker.get_order_statistic(1), 0);
  EXPECT_EQ(tracker.get_order_statistic(2), 0);
  EXPECT_EQ(tracker.count_less(0), 0);
  EXPECT_EQ(tracker.count_less_or_equal(0), 2);
  EXPECT_EQ(tracker.get_order_statistic(3), 1);
  EXPECT_EQ(tracker.count_less(1), 2);
  EXPECT_EQ(tracker.count_less_or_equal(1), 3);
}

class TestCI : public ConfidenceInterval {
  double radius(const int, const double) const override {
    return 0.1;
  }
};

TEST(QuantileCITrackerTest, BasicUse) {
  auto make_os_tracker = []() {return std::make_unique<TreeTracker>();};
  QuantileCITracker tracker(2, 0.59, 0.1, make_os_tracker,
                            std::make_unique<TestCI>(),
                            std::make_unique<TestCI>(),
                            std::make_unique<TestCI>(),
                            std::make_unique<TestCI>());
  EXPECT_EQ(tracker.num_arms(), 2);

  for (int i = 1; i <= 10; i++) {
    tracker.insert(0, i);
  }

  EXPECT_EQ(tracker.point_estimate(0, false), 6);
  EXPECT_EQ(tracker.upper_bound(0, .05, false), 7);
  EXPECT_EQ(tracker.lower_bound(0, .05, false), 5);
  EXPECT_EQ(tracker.point_estimate(0, true), 7);
  EXPECT_EQ(tracker.upper_bound(0, .05, true), 8);
  EXPECT_EQ(tracker.lower_bound(0, .05, true), 6);

  for (int i = 11; i <= 100; i++) {
    tracker.insert(0, i);
  }

  EXPECT_EQ(tracker.point_estimate(0, false), 60);
  EXPECT_EQ(tracker.upper_bound(0, .05, false), 70);
  EXPECT_EQ(tracker.point_estimate(0, true), 70);
}

struct CIValues {
  double point_estimate = 0;
  double lower_bound = -1;
  double upper_bound = 1;
};

class FakeCITracker : public QuantileCIInterface {
  public:

  void insert(const int arm_index, const double value) override {
    values_[arm_index].push_back(value);
  }

  double lower_bound(const int arm_index, const double error_rate,
                     const bool add_epsilon) const override {
    return (add_epsilon ? p_epsilon_ci_values : p_ci_values)[arm_index]
        .lower_bound;
  }
  double upper_bound(const int arm_index, const double error_rate,
                     const bool add_epsilon) const override {
    return (add_epsilon ? p_epsilon_ci_values : p_ci_values)[arm_index]
        .upper_bound;
  }
  double point_estimate(const int arm_index, const bool add_epsilon) const
      override {
    return (add_epsilon ? p_epsilon_ci_values : p_ci_values)[arm_index]
        .point_estimate;
  }
  int num_arms() const override {return 3;}

  std::vector<double> values_[3];
  CIValues p_ci_values[3];
  CIValues p_epsilon_ci_values[3];
};

TEST(QpacAgent, BasicUse) {
  auto tracker_unique_ptr = std::make_unique<FakeCITracker>();
  FakeCITracker* tracker = tracker_unique_ptr.get();
  QpacAgent agent(0.05, std::move(tracker_unique_ptr));

  for (int i = 0; i < 3; i++) {
    int arm = agent.get_arm_to_sample();
    EXPECT_EQ(agent.update(arm, arm), Agent::NO_ARM_SELECTED);
  }
  for (int arm = 0; arm < 3; arm++) {
    EXPECT_EQ(tracker->values_[arm].size(), 1);
    EXPECT_EQ(tracker->values_[arm][0], arm);
  }

  // eliminate arm 0
  tracker->p_epsilon_ci_values[0].upper_bound = -2;
  for (int i = 0; i < 3; i++) {
    int arm = agent.get_arm_to_sample();
    EXPECT_EQ(agent.update(arm, arm), Agent::NO_ARM_SELECTED);
  }
  EXPECT_EQ(tracker->values_[0].size(), 2);
  for (int i = 0; i < 4; i++) {
    int arm = agent.get_arm_to_sample();
    EXPECT_EQ(agent.update(arm, arm), Agent::NO_ARM_SELECTED);
  }
  EXPECT_EQ(tracker->values_[0].size(), 2);
  EXPECT_EQ(tracker->values_[1].size(), 4);
  EXPECT_EQ(tracker->values_[2].size(), 4);

  // choose arm 2
  tracker->p_epsilon_ci_values[2].lower_bound = 2;
  int arm = agent.get_arm_to_sample();
  EXPECT_EQ(agent.update(arm, arm), Agent::NO_ARM_SELECTED);
  arm = agent.get_arm_to_sample();
  EXPECT_EQ(agent.update(arm, arm), 2);
}

class FakeOrderStatisticTracker : public OrderStatisticTracker {
  public:

  void insert(const double value) override {
    inserted_values_.push_back(value);
  }

  double get_order_statistic(const int order_index) const
      override {
    auto search = order_stats_.find(order_index);
    if (search != order_stats_.end()) {
      return search->second;
    } else {
      return 0;
    }
  }

  int count_less_or_equal(const double value) const
      override {
    assert(false);
    return -1;
  }

  int count_less(const double value) const override {
    assert(false);
    return -1;
  }

  int size() const override {
    return inserted_values_.size();
  }

  std::vector<double> inserted_values_;
  std::unordered_map<int,double> order_stats_;
};

int m_k(int num_samples) {
  return floor(.5 * num_samples - sqrt(3 * .5 * num_samples * 24.26684)) + 1;
}

TEST(DoubledMaxQAgent, BasicUse) {
  std::vector<FakeOrderStatisticTracker*> trackers;
  auto make_os_tracker = [&trackers]() {
    auto tracker_unique_ptr = std::make_unique<FakeOrderStatisticTracker>();
    trackers.push_back(tracker_unique_ptr.get());
    return tracker_unique_ptr;
  };
  DoubledMaxQAgent agent(3, 0.05, 0.1, 0.5, make_os_tracker);

  // 6*log(3*log(-20*.5*log(.05)/.1^2, 2)) - log(.05)
  // L_D = 24.26684

  // floor(3*24.26684 / .5) + 1
  const int N_0 = 146;

  // 10 * .5 * 24.26684 / .1^2
  // stopping sample size = 12133.42

  // make arm 2 the highest after the initial round
  trackers[2]->order_stats_[N_0 - m_k(N_0) + 1] = 1;

  // initial sampling N_0 times from each arm
  for (int i = 0; i < N_0 * 3; i++) {
    int arm = agent.get_arm_to_sample();
    EXPECT_EQ(agent.update(arm, arm), Agent::NO_ARM_SELECTED);
  }
  for (int arm = 0; arm < 3; arm++) {
    EXPECT_EQ(trackers[arm]->size(), N_0);
    EXPECT_EQ(trackers[arm]->inserted_values_[0], arm);
  }
  EXPECT_EQ(agent.get_arm_to_sample(), 2);

  // another N_0 samples from arm 2
  trackers[2]->order_stats_[2 * N_0 - m_k(2 * N_0) + 1] = 1;
  for (int i = 0; i < N_0; i++) {
    EXPECT_EQ(agent.get_arm_to_sample(), 2);
    EXPECT_EQ(agent.update(2, 0), Agent::NO_ARM_SELECTED);
  }
  EXPECT_EQ(trackers[2]->size(), 2 * N_0);
  EXPECT_EQ(agent.get_arm_to_sample(), 2);

  // another 2*N_0 samples from arm 2, then switch to arm 1
  trackers[1]->order_stats_[N_0 - m_k(N_0) + 1] = 2;
  for (int i = 0; i < 2 * N_0; i++) {
    EXPECT_EQ(agent.get_arm_to_sample(), 2);
    EXPECT_EQ(agent.update(2, 0), Agent::NO_ARM_SELECTED);
  }
  EXPECT_EQ(agent.get_arm_to_sample(), 1);

  // take N_0 samples from arm 1
  trackers[1]->order_stats_[2 * N_0 - m_k(2*N_0) + 1] = 2;
  for (int i = 0; i < N_0; i++) {
    EXPECT_EQ(agent.get_arm_to_sample(), 1);
    EXPECT_EQ(agent.update(1, 0), Agent::NO_ARM_SELECTED);
  }
  EXPECT_EQ(trackers[2]->size(), 4 * N_0);
  EXPECT_EQ(trackers[1]->size(), 2 * N_0);

  // log(12134 / 146, 2) = 6.4
  // so sample seven rounds total from arm 1, hence 2^7 * 146 = 18688 samples
  // before stopping
  for (int k = 2; k <= 7; k++) {
    trackers[1]->order_stats_[pow(2, k) * N_0 - m_k(pow(2, k) * N_0) + 1] = 2;
  }
  for (int i = 0; i < 18688 - 2 * N_0 - 1; i++) {
    EXPECT_EQ(agent.get_arm_to_sample(), 1);
    EXPECT_EQ(agent.update(1, 0), Agent::NO_ARM_SELECTED);
  }
  EXPECT_EQ(agent.get_arm_to_sample(), 1);
  EXPECT_EQ(agent.update(1, 0), 1);
}

TEST(LucbAgent, BasicUse) {
  auto tracker_unique_ptr = std::make_unique<FakeCITracker>();
  FakeCITracker* tracker = tracker_unique_ptr.get();
  LucbAgent agent(0.05, std::move(tracker_unique_ptr));

  for (int i = 0; i < 3; i++) {
    int arm = agent.get_arm_to_sample();
    EXPECT_EQ(agent.update(arm, arm), Agent::NO_ARM_SELECTED);
  }
  for (int arm = 0; arm < 3; arm++) {
    EXPECT_EQ(tracker->values_[arm].size(), 1);
    EXPECT_EQ(tracker->values_[arm][0], arm);
  }

  // best arm = 1, best competitor = 2
  tracker->p_epsilon_ci_values[1].lower_bound = 2;
  tracker->p_ci_values[2].upper_bound = 3;

  for (int i = 0; i < 2; i++) {
    int arm = agent.get_arm_to_sample();
    EXPECT_EQ(agent.update(arm, 0), Agent::NO_ARM_SELECTED);
  }
  EXPECT_EQ(tracker->values_[0].size(), 1);
  EXPECT_EQ(tracker->values_[1].size(), 2);
  EXPECT_EQ(tracker->values_[2].size(), 2);

  // now stop with best arm 2
  tracker->p_epsilon_ci_values[2].lower_bound = 4;
  tracker->p_ci_values[1].upper_bound = 3;
  int arm = agent.get_arm_to_sample();
  EXPECT_TRUE(arm == 1 || arm == 2);
  EXPECT_EQ(agent.update(arm, 0), 2);
}

} // namespace
