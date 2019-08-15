#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

//#include "gperftools/profiler.h"

#include "quantile_bai.h"

double BanditModel::sample_arm(const int arm_index, std::mt19937_64& rng) const
{
  assert(0 <= arm_index && arm_index < num_arms());
  return arm_generator_(locations_[arm_index], rng);
}

void BanditModel::dump() const {
  printf("Locations");
  for (double location : locations_) {
    printf(" %.2f", location);
  }
  printf("\n");
}

TreeNode* const TreeNode::NULL_NODE = new TreeNode();

TreeNode* TreeNode::rotate_left() {
  assert(!is_null());
  assert(!right_->is_null());
  TreeNode* new_root = right_;
  right_ = new_root->left_;
  right_->parent_ = this;
  new_root->left_ = this;
  parent_ = new_root;
  refresh();
  new_root->refresh();
  return new_root;
}

TreeNode* TreeNode::rotate_right() {
  assert(!is_null());
  assert(!left_->is_null());
  TreeNode* new_root = left_;
  left_ = new_root->right_;
  left_->parent_ = this;
  new_root->right_ = this;
  parent_ = new_root;
  refresh();
  new_root->refresh();
  return new_root;
}

TreeNode* TreeNode::rotate_right_left() {
  assert(!is_null());
  assert(!right_->is_null());
  assert(!right_->left_->is_null());
  TreeNode* new_root = right_->left_;
  right_->left_ = new_root->right_;
  right_->left_->parent_ = right_;
  new_root->right_ = right_;
  new_root->right_->parent_ = new_root;
  right_ = new_root->left_;
  right_->parent_ = this;
  new_root->left_ = this;
  parent_ = new_root;
  new_root->left_->refresh();
  new_root->right_->refresh();
  new_root->refresh();
  return new_root;
}

TreeNode* TreeNode::rotate_left_right() {
  assert(!is_null());
  assert(!left_->is_null());
  assert(!left_->right_->is_null());
  TreeNode* new_root = left_->right_;
  left_->right_ = new_root->left_;
  left_->right_->parent_ = left_;
  new_root->left_ = left_;
  new_root->left_->parent_ = new_root;
  left_ = new_root->right_;
  left_->parent_ = this;
  new_root->right_ = this;
  parent_ = new_root;
  new_root->right_->refresh();
  new_root->left_->refresh();
  new_root->refresh();
  return new_root;
}

TreeNode* TreeNode::insert(const double new_value) {
  if (is_null()) {
    return new TreeNode(new_value);
  } else if (new_value <= value_) {
    left_ = left_->insert(new_value);
    left_->parent_ = this;
  } else {
    right_ = right_->insert(new_value);
    right_->parent_ = this;
  }

  refresh();

  if (balance_factor() < -1) {
    if (left_->balance_factor() > 0) {
      return rotate_left_right();
    } else {
      return rotate_right();
    }
  } else if (balance_factor() > 1) {
    if (right_->balance_factor() < 0) {
      return rotate_right_left();
    } else {
      return rotate_left();
    }
  } else {
    return this;
  }
}

double TreeNode::search(const int target_count, const int count_leq_above)
    const {
  assert(!is_null());
  int count_leq = count_leq_above + 1 + left_->subtree_count_;
  if (count_leq == target_count) {
    return value_;
  } else if (count_leq < target_count) {
    return right_->search(target_count, count_leq);
  } else {
    return left_->search(target_count, count_leq_above);
  }
}

double TreeNode::count_less(const double target_value,
                            const int count_less_above) const {
  if (is_null()) {
    return count_less_above;
  }

  if (value_ < target_value) {
    int count_less = count_less_above + 1 + left_->subtree_count_;
    return right_->count_less(target_value, count_less);
  } else { // value_ >= target_value
    return left_->count_less(target_value, count_less_above);
  }
}

double TreeNode::count_less_or_equal(const double target_value,
                                     const int count_leq_above) const {
  if (is_null()) {
    return count_leq_above;
  }

  if (value_ <= target_value) {
    int count_leq = count_leq_above + 1 + left_->subtree_count_;
    return right_->count_less_or_equal(target_value, count_leq);
  } else { // value_ > target_value
    return left_->count_less_or_equal(target_value, count_leq_above);
  }
}

const TreeNode* TreeNode::next() const {
  if (!right_->is_null()) {
    return right_->first_in_subtree();
  } else {
    const TreeNode* node = this;
    while (node != NULL) {
      if (node->parent_ != NULL && node->parent_->left_ == node) {
        return node->parent_;
      } else {
        node = node->parent_;
      }
    }
    return NULL;
  }
}

const TreeNode* TreeNode::first_in_subtree() const {
  if (left_->is_null()) {
    return this;
  } else {
    return left_->first_in_subtree();
  }
}

void TreeTracker::insert(const double value) {
  root_ = root_->insert(value);
}

double TreeTracker::get_order_statistic(const int order_index) const {
  assert(1 <= order_index && order_index <= size());
  assert(!root_->is_null());
  return root_->search(order_index, 0);
}


int TreeTracker::size() const {
  return root_->subtree_count_;
}

int TreeTracker::count_less(const double value) const {
  return root_->count_less(value, 0);
}

int TreeTracker::count_less_or_equal(const double value)
    const {
  return root_->count_less_or_equal(value, 0);
}

void QuantileCITracker::insert(const int arm_index, const double value) {
  assert(0 <= arm_index && arm_index < num_arms());
  os_trackers_[arm_index]->insert(value);
  p_ci_cache_[arm_index].invalidate();
  p_epsilon_ci_cache_[arm_index].invalidate();
}

double QuantileCITracker::lower_bound(
    const int arm_index, const double error_rate, const bool add_epsilon) const
{
  assert(0 <= arm_index && arm_index < num_arms());
  assert(os_trackers_[arm_index]->size() > 0);
  double p = quantile_p_;
  const QuantileCI* cached_ci = &p_ci_cache_[arm_index];
  const ConfidenceInterval* ci = p_lower_ci_.get();
  if (add_epsilon) {
    p += epsilon_;
    cached_ci = &p_epsilon_ci_cache_[arm_index];
    ci = p_epsilon_lower_ci_.get();
  }

  return cached_ci->lower_.get([this, arm_index, p, error_rate, ci]() {
    double lower_ci_radius = ci->radius(
        os_trackers_[arm_index]->size(), error_rate);
    return get_quantile(arm_index, p - lower_ci_radius);
  });
}

double QuantileCITracker::upper_bound(
    const int arm_index, const double error_rate, const bool add_epsilon) const
{
  assert(0 <= arm_index && arm_index < num_arms());
  assert(os_trackers_[arm_index]->size() > 0);
  double p = quantile_p_;
  const QuantileCI* cached_ci = &p_ci_cache_[arm_index];
  const ConfidenceInterval* ci = p_upper_ci_.get();
  if (add_epsilon) {
    p += epsilon_;
    cached_ci = &p_epsilon_ci_cache_[arm_index];
    ci = p_epsilon_upper_ci_.get();
  }

  return cached_ci->upper_.get([this, arm_index, p, ci, error_rate]() {
    double upper_ci_radius = ci->radius(
        os_trackers_[arm_index]->size(), error_rate);
    return get_quantile(arm_index, p + upper_ci_radius);
  });
}

double QuantileCITracker::point_estimate(const int arm_index,
                                         const bool add_epsilon) const {
  assert(0 <= arm_index && arm_index < num_arms());
  assert(os_trackers_[arm_index]->size() > 0);
  double p = quantile_p_;
  const QuantileCI* cached_ci = &p_ci_cache_[arm_index];
  if (add_epsilon) {
    p += epsilon_;
    cached_ci = &p_epsilon_ci_cache_[arm_index];
  }

  return cached_ci->point_estimate_.get([this, arm_index, p]() {
    return get_quantile(arm_index, p);
  });
}

double QuantileCITracker::get_quantile(const int arm_index, const double p)
    const {
  assert(0 <= arm_index && arm_index < num_arms());
  double target_index = p * os_trackers_[arm_index]->size();
  int target_count = floor(target_index) + 1;
  // include some thresholds on p to handle numerical error
  if (p < 1e-10 || target_count < 1) {
    return -std::numeric_limits<double>::infinity();
  } else if (p > 1 - 1e-10 || target_count > os_trackers_[arm_index]->size()) {
    return std::numeric_limits<double>::infinity();
  } else {
    return os_trackers_[arm_index]->get_order_statistic(target_count);
  }
}

const int Agent::NO_ARM_SELECTED = -1;

int QpacAgent::get_arm_to_sample() const {
  assert(next_arm_iter_ != active_arms_.end());
  return *next_arm_iter_;
}

int QpacAgent::update(const int arm_pulled, const double value) {
  ci_tracker_->insert(arm_pulled, value);
  next_arm_iter_++;
  if (next_arm_iter_ != active_arms_.end()) {
    return NO_ARM_SELECTED;
  }

  double max_upper_bound = -std::numeric_limits<double>::infinity();
  double second_max_upper_bound = -std::numeric_limits<double>::infinity();
  int max_ub_arm;
  double max_lower_bound = -std::numeric_limits<double>::infinity();
  for (int arm_index : active_arms_) {
    double p_lower = ci_tracker_->lower_bound(arm_index, error_rate_, false);
    double p_upper = ci_tracker_->upper_bound(arm_index, error_rate_, false);
    if (p_upper > max_upper_bound) {
      second_max_upper_bound = max_upper_bound;
      max_upper_bound = p_upper;
      max_ub_arm = arm_index;
    } else if (p_upper > second_max_upper_bound) {
      second_max_upper_bound = p_upper;
    }
    if (p_lower > max_lower_bound) {
      max_lower_bound = p_lower;
    }
  }

  for (auto arm_iter = active_arms_.begin(); arm_iter != active_arms_.end(); ) {
    double p_epsilon_lower = ci_tracker_->lower_bound(*arm_iter, error_rate_,
                                                      true);
    double p_epsilon_upper = ci_tracker_->upper_bound(*arm_iter, error_rate_,
                                                      true);
    double other_arms_max_upper_bound =
        *arm_iter == max_ub_arm ? second_max_upper_bound : max_upper_bound;
    if (other_arms_max_upper_bound < p_epsilon_lower) {
      return *arm_iter;
    }
    if (p_epsilon_upper < max_lower_bound) {
      arm_iter = active_arms_.erase(arm_iter);
    } else {
      arm_iter++;
    }
  }

  if (active_arms_.size() == 1) {
    return *active_arms_.begin();
  }

  next_arm_iter_ = active_arms_.begin();
  return NO_ARM_SELECTED;
}

int DoubledMaxQAgent::get_arm_to_sample() const {
  return next_arm_to_sample_;
}

int DoubledMaxQAgent::update(const int arm_pulled, const double value) {
  assert(arm_pulled == next_arm_to_sample_);
  os_trackers_[arm_pulled]->insert(value);
  samples_remaining_--;
  if (samples_remaining_ > 0) {
    return NO_ARM_SELECTED;
  }
  if (!is_initial_sampling_done_) {
    next_arm_to_sample_++;
    if (next_arm_to_sample_ == os_trackers_.size()) {
      is_initial_sampling_done_ = true;
    } else {
      samples_remaining_ = N_0_;
      return NO_ARM_SELECTED;
    }
  }

  int best_arm = -1;
  double best_value = -std::numeric_limits<double>::infinity();
  for (int arm = 0; arm < os_trackers_.size(); arm++) {
    int m_k = floor(
        (1 - quantile_p_) * os_trackers_[arm]->size()
        - sqrt(3 * (1 - quantile_p_) * os_trackers_[arm]->size() * L_D_))
        + 1;
    int order = os_trackers_[arm]->size() - m_k + 1;
    double V_k = os_trackers_[arm]->get_order_statistic(order);
    if (V_k > best_value) {
      best_value = V_k;
      best_arm = arm;
    }
  }

  if (os_trackers_[best_arm]->size() > stop_sample_size_) {
    return best_arm;
  } else {
    next_arm_to_sample_ = best_arm;
    samples_remaining_ = os_trackers_[best_arm]->size();
    return NO_ARM_SELECTED;
  }
}

int LucbAgent::best_arm() const {
  int best_index = 0;
  double max_p_epsilon_lower = -std::numeric_limits<double>::infinity();
  for (int arm = 0; arm < tracker_->num_arms(); arm++) {
    double p_epsilon_lower = tracker_->lower_bound(arm, error_rate_, true);
    if (p_epsilon_lower > max_p_epsilon_lower) {
      max_p_epsilon_lower = p_epsilon_lower;
      best_index = arm;
    }
  }
  return best_index;
}

int LucbAgent::best_competitor() const {
  int best_arm_index = best_arm();
  int best_competitor_index = -1;
  double best_competitor_ucb = -std::numeric_limits<double>::infinity();
  for (int arm = 0; arm < tracker_->num_arms(); arm++) {
    if (arm == best_arm_index) continue;
    double ucb = tracker_->upper_bound(arm, error_rate_, false);
    if (ucb > best_competitor_ucb) {
      best_competitor_ucb = ucb;
      best_competitor_index = arm;
    }
  }
  assert(best_competitor_index > -1);
  return best_competitor_index;
}

bool LucbAgent::can_stop() const {
  double best_arm_lcb = tracker_->lower_bound(best_arm(), error_rate_, true);
  double best_competitor_ucb = tracker_->upper_bound(best_competitor(),
                                                     error_rate_, false);
  return best_arm_lcb >= best_competitor_ucb;
}

int LucbAgent::get_arm_to_sample() const {
  if (initial_sampling_index_ >= 0) {
    return initial_sampling_index_;
  } else if (sample_best_arm_next_) {
    return best_arm();
  } else {
    return best_competitor();
  }
}

int LucbAgent::update(const int arm_pulled, const double value) {
  tracker_->insert(arm_pulled, value);

  if (initial_sampling_index_ >= 0) {
    initial_sampling_index_--;
    if (initial_sampling_index_ >= 0) {
      return NO_ARM_SELECTED;
    }
  } else {
    sample_best_arm_next_ = !sample_best_arm_next_;
  }

  if (can_stop()) {
    return best_arm();
  } else {
    return NO_ARM_SELECTED;
  }
}

int ABTestAgent::get_arm_to_sample() const {
  return next_arm_;
}

int ABTestAgent::update(const int arm_pulled, const double value) {
  os_trackers_[arm_pulled]->insert(value);
  os_trackers_[2]->insert(value);
  next_arm_ = !next_arm_;
  if (std::min(os_trackers_[0]->size(), os_trackers_[1]->size()) < 1) {
    return NO_ARM_SELECTED;
  }
  else if (ab_test_.p_value() <= error_rate_) {
    return best_arm();
  } else {
    return NO_ARM_SELECTED;
  }
}

double ABTestAgent::empirical_quantile(const int arm) const {
  int position = floor(os_trackers_[arm]->size() * quantile_p_) + 1;
  return os_trackers_[arm]->get_order_statistic(position);
}

double ABTestAgent::best_arm() const {
  return empirical_quantile(1) > empirical_quantile(0) ? 1 : 0;
}

ResultMap simulate_one_run(const BanditModel& bandit, AgentMap agents,
                           const int max_rounds, const bool is_ab_test,
                           std::mt19937_64& rng) {
  std::unordered_set<std::string> remaining_agents;
  for (auto name_agent_pair : agents) {
    remaining_agents.insert(name_agent_pair.first);
  }

  int round = 0;
  ResultMap results;
  std::unordered_map<int, double> arm_values;
  int ab_arm = 0;
  while (!remaining_agents.empty() && round < max_rounds) {
    round++;
    arm_values.clear();
    auto name_iter = remaining_agents.begin();
    while (name_iter != remaining_agents.end()) {
      std::shared_ptr<Agent> agent = agents[*name_iter];
      int arm_index = is_ab_test ? ab_arm : agent->get_arm_to_sample();
      if (arm_values.count(arm_index) == 0) {
        arm_values[arm_index] = bandit.sample_arm(arm_index, rng);
      }
      int chosen_arm = agent->update(arm_index, arm_values[arm_index]);
      if (chosen_arm != Agent::NO_ARM_SELECTED) {
        assert(results.count(*name_iter) == 0);
        RunResult result = {chosen_arm, round};
        results[*name_iter] = result;
        name_iter = remaining_agents.erase(name_iter);
      } else {
        name_iter++;
      }
    }
    ab_arm = !ab_arm;
  }

  RunResult not_finished_result = {-1, round};
  for (auto name : remaining_agents) {
    results[name] = not_finished_result;
  }

  return results;
}

void simulate_many_walks(const int thread_id,
                         const SimulationConfig* const config,
                         const bool* const stop_all_threads,
                         std::vector<ResultMap>* all_results) {
  std::mt19937_64 rng;
  const int reps_per_thread = config->num_replications_ / config->num_threads_;
  const int start_index = thread_id * reps_per_thread;
  const int target_min_sample_size = 1000;

  auto make_szorenyi = [](const double) {
    return std::make_unique<SzorenyiCI>();
  };
  auto make_bb = [config, target_min_sample_size](const double p) {
    return std::make_unique<BetaBinomialCI>(
        p, target_min_sample_size, config->error_rate_ / config->num_arms());
  };
  auto make_stitched = [target_min_sample_size](const double p) {
    return std::make_unique<StitchedCI>(p, target_min_sample_size, 2.04, 1.4);
  };

  auto make_ci_tracker = [config](
      std::function<std::unique_ptr<const ConfidenceInterval>(double)> make_ci)
  {
    return std::make_unique<QuantileCITracker>(
        config->num_arms(), config->quantile_p_, config->epsilon_,
        config->make_os_tracker_,
        make_ci(1 - config->quantile_p_),
        make_ci(config->quantile_p_),
        make_ci(1 - config->quantile_p_ - config->epsilon_),
        make_ci(config->quantile_p_ + config->epsilon_));
  };

  //ProfilerStart("/Users/steve/temp/profiler.out");
  for (int replication_index = start_index;
       replication_index < start_index + reps_per_thread;
       replication_index++) {
    rng.seed(293887823 + replication_index);
    if (*stop_all_threads) break;

    AgentMap agents;
    if (config->ab_test_enabled_) {
      assert(config->num_arms() == 2);
      agents["LUCB BB"] = std::make_shared<LucbAgent>(
          config->error_rate_, make_ci_tracker(make_bb));
      agents["ABTest"] = std::make_shared<ABTestAgent>(
          config->error_rate_, config->quantile_p_, target_min_sample_size,
          config->make_os_tracker_);
    } else {
      auto szorenyi_ci = std::make_shared<SzorenyiCI>();
      agents["QPAC"] = std::make_shared<QpacAgent>(
          config->error_rate_, make_ci_tracker(make_szorenyi));
      agents["Doubled Max-Q"] = std::make_shared<DoubledMaxQAgent>(
          config->num_arms(), config->error_rate_, config->epsilon_,
          config->quantile_p_, config->make_os_tracker_);
      agents["LUCB DKW"] = std::make_shared<LucbAgent>(
          config->error_rate_, make_ci_tracker(make_szorenyi));
      agents["LUCB Stitched"] = std::make_shared<LucbAgent>(
          config->error_rate_, make_ci_tracker(make_stitched));
      agents["QPAC BB"] = std::make_shared<QpacAgent>(
          config->error_rate_, make_ci_tracker(make_bb));
      agents["LUCB BB"] = std::make_shared<LucbAgent>(
          config->error_rate_, make_ci_tracker(make_bb));
    }

    (*all_results)[replication_index] = simulate_one_run(
        config->model_, agents, config->max_rounds_, config->ab_test_enabled_,
        rng);

    if (thread_id == 0 && ((replication_index + 1) % 10) == 0) {
      fprintf(stderr, "%d\r", (replication_index + 1) * config->num_threads_);
      fflush(stderr);
    }
  }
  //ProfilerStop();
}
