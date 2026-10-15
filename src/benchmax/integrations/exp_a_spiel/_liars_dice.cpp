// BenchMAX implementation, not part of the vendored exp-a-spiel source.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace py = pybind11;

namespace {
template <int Dice, int Faces> class LiarsDice {
  static_assert(Dice == 1 || Dice == 2);
  static constexpr int Bids = 2 * Dice * Faces;
  static_assert(Bids <= 20);
  static constexpr int Hands = Dice == 1 ? Faces : Faces * (Faces + 1) / 2;
  using HandValues = std::array<double, Hands>;
  using Reach = std::array<HandValues, 2>;
  struct Values {
    Reach br{};
    HandValues ev{}; // Player 0's value, excluding player 0's reach.
  };

  std::array<std::array<int, Dice>, Hands> dice_{};
  HandValues chance_{};
  std::array<std::vector<uint32_t>, 2> histories_;
  std::vector<size_t> offsets_;
  std::array<size_t, 2> sizes_{};
  std::array<std::array<HandValues, Hands>, Bids> caller_payoffs_{};

  static int check_player(int player) {
    if (player != 0 && player != 1)
      throw py::value_error("player must be 0 or 1");
    return player;
  }

  static int first_action(uint32_t history) {
    return std::bit_width(history);
  }

  static int action_count(uint32_t history) {
    return Bids - first_action(history) + (history != 0);
  }

  void check_range(int player, py::ssize_t start, py::ssize_t stop) const {
    check_player(player);
    if (start < 0 || stop < start || stop > num_information_states(player))
      throw py::value_error("invalid information-state batch range");
  }

  static void check_array(const py::array &array) {
    if (!array.dtype().is(py::dtype::of<double>()))
      throw py::type_error("strategy/probabilities must have dtype float64");
    if (!(array.flags() & py::array::c_style))
      throw py::type_error("strategy/probabilities must be C-contiguous");
  }

  const double *strategy_data(int player, const py::array &strategy) const {
    check_player(player);
    check_array(strategy);
    if (strategy.ndim() != 1 || static_cast<size_t>(strategy.size()) != sizes_[player])
      throw py::value_error("strategy has the wrong packed shape");
    return static_cast<const double *>(strategy.data());
  }

  void validate_strategy(int player, const double *strategy) const {
    for (uint32_t history : histories_[player]) {
      const int count = action_count(history);
      for (int hand = 0; hand < Hands; ++hand) {
        double sum = 0;
        for (int action = 0; action < count; ++action) {
          const double value = strategy[offsets_[history] + action * Hands + hand];
          if (!std::isfinite(value) || value < 0 || value > 1)
            throw py::value_error("not a valid strategy");
          sum += value;
        }
        if (std::abs(sum - 1) > 1e-6)
          throw py::value_error("not a valid strategy");
      }
    }
  }

  // Counterfactual reach includes chance and the fixed player's actions only.
  // Maximization is performed after summing over the opponent's hidden hand.
  // In particular, zero own reach must never prune a best-response branch.
  Values visit(uint32_t history, int player, const Reach &reach,
               const std::array<const double *, 2> &strategies) const {
    Values result;
    const int other = 1 - player;
    const int first = first_action(history);
    const double *policy = strategies[player] + offsets_[history];
    result.br[player].fill(-std::numeric_limits<double>::infinity());

    if (history) {
      const double *call = policy + (Bids - first) * Hands;
      for (int own = 0; own < Hands; ++own) {
        double call_value = 0;
        for (int opp = 0; opp < Hands; ++opp) {
          const double payoff = caller_payoffs_[first - 1][own][opp];
          call_value += payoff * reach[other][opp];
          result.br[other][opp] -= payoff * reach[player][own] * call[own];
        }
        result.br[player][own] = call_value;
        if (player == 0)
          result.ev[own] = call_value * call[own];
      }
      if (player == 1)
        result.ev = result.br[0];
    }

    for (int action = first; action < Bids; ++action) {
      const double *probability = policy + (action - first) * Hands;
      Reach child_reach = reach;
      for (int hand = 0; hand < Hands; ++hand)
        child_reach[player][hand] *= probability[hand];
      const Values child = visit(history | (1u << action), other,
                                 child_reach, strategies);
      for (int hand = 0; hand < Hands; ++hand) {
        result.br[player][hand] =
            std::max(result.br[player][hand], child.br[player][hand]);
        result.br[other][hand] += child.br[other][hand];
        result.ev[hand] += child.ev[hand] * (player == 0 ? probability[hand] : 1);
      }
    }
    return result;
  }

public:
  static constexpr int NumActions = Bids + 1;
  static constexpr int InformationStateSize = 2 + Dice * Faces + NumActions;

  LiarsDice() : offsets_(1u << Bids) {
    int hand = 0;
    for (int first = 1; first <= Faces; ++first) {
      if constexpr (Dice == 1) {
        dice_[hand] = {first};
        chance_[hand++] = 1.0 / Faces;
      } else {
        for (int second = first; second <= Faces; ++second) {
          dice_[hand] = {first, second};
          chance_[hand++] = (first == second ? 1.0 : 2.0) / (Faces * Faces);
        }
      }
    }
    for (uint32_t history = 0; history < (1u << Bids); ++history) {
      const int player = std::popcount(history) % 2;
      histories_[player].push_back(history);
      offsets_[history] = sizes_[player];
      sizes_[player] += action_count(history) * Hands;
    }
    for (int bid = 0; bid < Bids; ++bid) {
      const int quantity = bid / Faces + 1;
      const int face = bid % Faces + 1;
      for (int own = 0; own < Hands; ++own) {
        for (int opp = 0; opp < Hands; ++opp) {
          int matches = 0;
          for (int die : dice_[own])
            matches += die == face || die == Faces;
          for (int die : dice_[opp])
            matches += die == face || die == Faces;
          caller_payoffs_[bid][own][opp] = matches < quantity ? 1.0 : -1.0;
        }
      }
    }
  }

  py::ssize_t num_information_states(int player) const {
    return histories_[check_player(player)].size() * Hands;
  }

  py::tuple policy_input_batch(int player, py::ssize_t start, py::ssize_t stop) const {
    check_range(player, start, stop);
    const py::ssize_t rows = stop - start;
    py::array_t<double> observations(std::array<py::ssize_t, 2>{rows, InformationStateSize});
    py::array_t<bool> masks(std::array<py::ssize_t, 2>{rows, NumActions});
    std::fill_n(observations.mutable_data(), observations.size(), 0.0);
    std::fill_n(masks.mutable_data(), masks.size(), false);
    auto obs = observations.template mutable_unchecked<2>();
    auto mask = masks.template mutable_unchecked<2>();
    for (py::ssize_t row = start; row < stop; ++row) {
      const auto history = histories_[player][row / Hands];
      const auto &dice = dice_[row % Hands];
      const auto out = row - start;
      obs(out, player) = 1;
      for (int die = 0; die < Dice; ++die)
        obs(out, 2 + die * Faces + dice[die] - 1) = 1;
      for (int bid = 0; bid < Bids; ++bid)
        obs(out, 2 + Dice * Faces + bid) = (history >> bid) & 1u;
      for (int action = first_action(history); action < Bids; ++action)
        mask(out, action) = true;
      mask(out, Bids) = history != 0;
    }
    return py::make_tuple(observations, masks);
  }

  py::array_t<double> empty_strategy(int player) const {
    return py::array_t<double>(sizes_[check_player(player)]);
  }

  void pack_policy_batch(int player, py::ssize_t start,
                         const py::array &probabilities, py::array &strategy) const {
    check_array(probabilities);
    if (probabilities.ndim() != 2 || probabilities.shape(1) != NumActions)
      throw py::value_error("probabilities have the wrong shape");
    check_range(player, start, start + probabilities.shape(0));
    strategy_data(player, strategy);
    auto *destination = static_cast<double *>(strategy.mutable_data());
    const auto *source = static_cast<const double *>(probabilities.data());
    for (py::ssize_t row = 0; row < probabilities.shape(0); ++row) {
      const auto history = histories_[player][(start + row) / Hands];
      const int hand = (start + row) % Hands;
      const int first = first_action(history);
      double sum = 0;
      for (int action = 0; action < NumActions; ++action) {
        const double value = source[row * NumActions + action];
        const bool legal = action >= first && (action < Bids || history != 0);
        if (!std::isfinite(value) || value < 0 || value > 1 || (!legal && value != 0))
          throw py::value_error("not a valid strategy");
        if (legal) {
          destination[offsets_[history] + (action - first) * Hands + hand] = value;
          sum += value;
        }
      }
      if (std::abs(sum - 1) > 1e-6)
        throw py::value_error("not a valid strategy");
    }
  }

  std::array<double, 3> evaluate(const py::array &strategy_0, const py::array &strategy_1) const {
    const std::array<const double *, 2> strategies = {
        strategy_data(0, strategy_0), strategy_data(1, strategy_1)};
    py::gil_scoped_release release;
    validate_strategy(0, strategies[0]);
    validate_strategy(1, strategies[1]);
    const Values values = visit(0, 0, Reach{chance_, chance_}, strategies);
    double ev = 0, br0 = 0, br1 = 0;
    for (int hand = 0; hand < Hands; ++hand) {
      ev += chance_[hand] * values.ev[hand];
      br0 += chance_[hand] * values.br[0][hand];
      br1 += chance_[hand] * values.br[1][hand];
    }
    // Match exp-a-spiel: policy 0's exploitability first (player 1's gain).
    return {ev, br1 + ev, br0 - ev};
  }
};

template <int Dice, int Faces>
void bind(py::module_ &module, const char *name) {
  using Evaluator = LiarsDice<Dice, Faces>;
  auto cls = py::class_<Evaluator>(module, name);
  py::dict parameters;
  parameters["players"] = 2;
  parameters["numdice"] = Dice;
  parameters["dice_sides"] = Faces;
  parameters["bidding_rule"] = "reset-face";
  py::dict game;
  game["spiel_name"] = "liars_dice";
  game["parameters"] = parameters;
  cls.attr("compatible_game") = game;
  cls.attr("num_actions") = Evaluator::NumActions;
  cls.attr("information_state_size") = Evaluator::InformationStateSize;
  cls.def(py::init<>())
      .def("num_information_states", &Evaluator::num_information_states, py::arg("player"))
      .def("policy_input_batch", &Evaluator::policy_input_batch,
           py::arg("player"), py::arg("start"), py::arg("stop"))
      .def("empty_strategy", &Evaluator::empty_strategy, py::arg("player"))
      .def("pack_policy_batch", &Evaluator::pack_policy_batch, py::arg("player"),
           py::arg("start"), py::arg("probabilities").noconvert(), py::arg("strategy").noconvert())
      .def("evaluate", [](const Evaluator &evaluator, const py::array &strategy_0,
                           const py::array &strategy_1) {
        const auto metrics = evaluator.evaluate(strategy_0, strategy_1);
        return py::make_tuple(metrics[0], metrics[1], metrics[2]);
      }, py::arg("strategy_player_0").noconvert(), py::arg("strategy_player_1").noconvert());
}
} // namespace

void register_liars_dice(py::module_ &module) {
  bind<1, 2>(module, "Ld1d2fEvaluator");
  bind<1, 3>(module, "Ld1d3fEvaluator");
  bind<1, 4>(module, "Ld1d4fEvaluator");
  bind<1, 5>(module, "Ld1d5fEvaluator");
  bind<1, 6>(module, "Ld1d6fEvaluator");
  bind<2, 2>(module, "Ld2d2fEvaluator");
  bind<2, 3>(module, "Ld2d3fEvaluator");
  bind<2, 4>(module, "Ld2d4fEvaluator");
  bind<2, 5>(module, "Ld2d5fEvaluator");
}
