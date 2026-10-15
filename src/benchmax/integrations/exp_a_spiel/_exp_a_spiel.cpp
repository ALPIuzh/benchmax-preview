#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <utility>

#include "dh_state.h"
#include "pttt_state.h"
#include "traverser.h"
#include "utils.h"

namespace py = pybind11;

void register_liars_dice(py::module_ &module);

namespace {
template <typename Success>
uint64_t decode_infoset(std::span<const double> observation,
                        size_t history_offset, Success success) {
  uint64_t infoset = 0;
  for (size_t turn = 0; turn < 9; ++turn) {
    size_t cell = 0;
    while (cell < 9 && observation[history_offset + 9 * turn + cell] == 0.0) {
      ++cell;
    }
    if (cell == 9) {
      break;
    }
    infoset = (infoset << 5) | ((cell + 1) << 1) | success(cell);
  }
  return infoset;
}

template <bool Abrupt>
uint64_t infoset_from_openspiel(const DhState<Abrupt> *, uint8_t player,
                                std::span<const double> observation) {
  using State = DhState<Abrupt>;
  const size_t own_stone = player == 0 ? State::kBlack - State::min_cell_state
                                      : State::kWhite - State::min_cell_state;
  return decode_infoset(
      observation, State::num_cells * State::cell_states, [&](size_t cell) {
        return observation[cell * State::cell_states + own_stone] != 0.0;
      });
}

uint64_t infoset_from_openspiel(const CornerDhState *, uint8_t player,
                                std::span<const double> observation) {
  return infoset_from_openspiel(static_cast<const DhState<false> *>(nullptr),
                                player, observation);
}

template <bool Abrupt>
uint64_t infoset_from_openspiel(const PtttState<Abrupt> *, uint8_t player,
                                std::span<const double> observation) {
  const size_t own_stone = player == 0 ? 18 : 9;
  return decode_infoset(observation, 27, [&](size_t cell) {
    return observation[own_stone + cell] != 0.0;
  });
}

uint8_t validate_player(py::ssize_t player) {
  if (player != 0 && player != 1) {
    throw py::value_error("player must be 0 or 1");
  }
  return static_cast<uint8_t>(player);
}

ConstRealBuf strategy_buffer(const py::array &strategy,
                             const Treeplex &treeplex,
                             const char *argument_name) {
  if (!strategy.dtype().is(py::dtype::of<Real>())) {
    throw py::type_error(std::string(argument_name) +
                         " must have dtype float64");
  }
  if ((strategy.flags() & py::array::c_style) == 0) {
    throw py::type_error(std::string(argument_name) +
                         " must be C-contiguous");
  }

  const auto rows = static_cast<py::ssize_t>(treeplex.num_infosets());
  if (strategy.ndim() != 2 || strategy.shape(0) != rows ||
      strategy.shape(1) != 9) {
    throw py::value_error(std::string(argument_name) + " must have shape (" +
                          std::to_string(rows) + ", 9)");
  }

  ConstRealBuf values(static_cast<const Real *>(strategy.data()),
                      static_cast<size_t>(strategy.size()));
  return values;
}

ConstRealBuf validate_strategy(const py::array &strategy,
                               const Treeplex &treeplex,
                               const char *argument_name) {
  ConstRealBuf values =
      strategy_buffer(strategy, treeplex, argument_name);
  if (!treeplex.is_valid_strategy(values)) {
    throw py::value_error(std::string(argument_name) +
                          " is not a valid strategy");
  }
  return values;
}

template <typename State>
void register_evaluator(py::module_ &module, const char *name,
                        py::dict compatible_game) {
  using Evaluator = Traverser<State>;

  auto evaluator = py::class_<Evaluator>(module, name);
  evaluator.attr("compatible_game") = std::move(compatible_game);
  evaluator.attr("num_actions") = 9;
  evaluator
      .def(py::init<>())
      .def_property_readonly_static(
          "information_state_size",
          [](py::object) { return State::OPENSPIEL_INFOSTATE_SIZE; })
      .def(
          "num_information_states",
          [](const Evaluator &evaluator, py::ssize_t player) {
            const uint8_t p = validate_player(player);
            return evaluator.treeplex[p]->num_infosets();
          },
          py::arg("player"))
      .def(
          "policy_input_batch",
          [](const Evaluator &evaluator, py::ssize_t player, py::ssize_t start,
             py::ssize_t stop) {
            const uint8_t p = validate_player(player);
            const py::ssize_t total = evaluator.treeplex[p]->num_infosets();
            if (start < 0) {
              throw py::value_error("start must be non-negative");
            }
            if (stop < start) {
              throw py::value_error("stop must be greater than or equal to start");
            }
            if (stop > total) {
              throw py::value_error(
                  "stop must not exceed num_information_states(player)");
            }

            const py::ssize_t rows = stop - start;
            py::array_t<Real> observations(
                std::array<py::ssize_t, 2>{
                    rows, State::OPENSPIEL_INFOSTATE_SIZE});
            py::array_t<bool> legal_action_mask(
                std::array<py::ssize_t, 2>{rows, 9});
            auto observation_values =
                observations.template mutable_unchecked<2>();
            auto mask_values =
                legal_action_mask.template mutable_unchecked<2>();

            std::array<bool, State::OPENSPIEL_INFOSTATE_SIZE> encoded;
            for (py::ssize_t row = start; row < stop; ++row) {
              evaluator.compute_openspiel_infostate(
                  p, row, std::span<bool>(encoded));
              const py::ssize_t output_row = row - start;
              for (py::ssize_t column = 0;
                   column < State::OPENSPIEL_INFOSTATE_SIZE; ++column) {
                observation_values(output_row, column) = encoded[column];
              }

              const uint32_t legal_actions =
                  evaluator.treeplex[p]->legal_actions[row];
              for (py::ssize_t action = 0; action < 9; ++action) {
                mask_values(output_row, action) =
                    legal_actions & (1u << action);
              }
            }
            return py::make_tuple(observations, legal_action_mask);
          },
          py::arg("player"), py::arg("start"), py::arg("stop"))
      .def(
          "cfr_treeplex",
          [](const Evaluator &evaluator, py::ssize_t player) {
            const uint8_t p = validate_player(player);
            const auto &treeplex = *evaluator.treeplex[p];
            const py::ssize_t rows = treeplex.num_infosets();
            py::ssize_t action_count = 0;
            for (const uint32_t mask : treeplex.legal_actions) {
              action_count += __builtin_popcount(mask);
            }

            py::array_t<uint32_t> actions(action_count);
            py::array_t<uint32_t> counts(rows);
            py::array_t<uint32_t> masks(rows);
            py::array_t<uint32_t> parents(rows);
            py::array_t<uint8_t> depths(rows);
            auto action_values = actions.template mutable_unchecked<1>();
            auto count_values = counts.template mutable_unchecked<1>();
            auto mask_values = masks.template mutable_unchecked<1>();
            auto parent_values = parents.template mutable_unchecked<1>();
            auto depth_values = depths.template mutable_unchecked<1>();

            py::ssize_t output = 0;
            for (py::ssize_t row = 0; row < rows; ++row) {
              const uint32_t mask = treeplex.legal_actions[row];
              count_values(row) = __builtin_popcount(mask);
              mask_values(row) = mask;
              if (row == 0) {
                depth_values(row) = 1;
                parent_values(row) = 0;
              } else {
                const uint32_t parent = treeplex.parent_index[row];
                depth_values(row) = depth_values(parent) + 1;
                parent_values(row) = parent * 9 +
                                     parent_action(treeplex.infoset_keys[row]);
              }
              for (uint8_t action = 0; action < 9; ++action) {
                if (mask & (1u << action)) {
                  action_values(output++) =
                      static_cast<uint32_t>(row) * 9 + action;
                }
              }
            }
            return py::make_tuple(actions, counts, masks, parents, depths);
          },
          py::arg("player"))
      .def(
          "cfr_gradient",
          [](Evaluator &evaluator, const py::array &strategy_player_0,
             const py::array &strategy_player_1, py::ssize_t player) {
            const uint8_t p = validate_player(player);
            const ConstRealBuf strategy_0 = validate_strategy(
                strategy_player_0, *evaluator.treeplex[0],
                "strategy_player_0");
            const ConstRealBuf strategy_1 = validate_strategy(
                strategy_player_1, *evaluator.treeplex[1],
                "strategy_player_1");
            {
              py::gil_scoped_release release;
              evaluator.compute_gradients({strategy_0, strategy_1});
            }
            auto &gradient = evaluator.gradients[p];
            return py::array_t<Real>(
                static_cast<py::ssize_t>(gradient.size()), &gradient[0],
                py::cast(&evaluator));
          },
          py::arg("strategy_player_0").noconvert(),
          py::arg("strategy_player_1").noconvert(), py::arg("player"))
      .def(
          "cfr_policy_batch",
          [](const Evaluator &evaluator, py::ssize_t player,
             const py::array_t<Real, py::array::c_style |
                                         py::array::forcecast> &observations,
             const py::array &strategy) {
            const uint8_t p = validate_player(player);
            if (observations.ndim() != 2 ||
                observations.shape(1) != State::OPENSPIEL_INFOSTATE_SIZE) {
              throw py::value_error(
                  "observations have the wrong information-state shape");
            }
            const ConstRealBuf strategy_values = strategy_buffer(
                strategy, *evaluator.treeplex[p], "strategy");
            const py::ssize_t rows = observations.shape(0);
            const py::ssize_t columns = observations.shape(1);
            py::array_t<Real> probabilities(
                std::array<py::ssize_t, 2>{rows, 9});
            auto output = probabilities.template mutable_unchecked<2>();
            {
              py::gil_scoped_release release;
              for (py::ssize_t row = 0; row < rows; ++row) {
                const uint64_t key = infoset_from_openspiel(
                    static_cast<const State *>(nullptr), p,
                    std::span<const Real>(observations.data(row, 0), columns));
                const uint32_t source =
                    evaluator.treeplex[p]->infosets.at(key).infoset_id * 9;
                for (uint8_t action = 0; action < 9; ++action) {
                  output(row, action) = strategy_values[source + action];
                }
              }
            }
            return probabilities;
          },
          py::arg("player"), py::arg("observations"),
          py::arg("strategy").noconvert())
      .def(
          "evaluate",
          [](Evaluator &evaluator, const py::array &strategy_player_0,
             const py::array &strategy_player_1) {
            const ConstRealBuf strategy_0 = validate_strategy(
                strategy_player_0, *evaluator.treeplex[0],
                "strategy_player_0");
            const ConstRealBuf strategy_1 = validate_strategy(
                strategy_player_1, *evaluator.treeplex[1],
                "strategy_player_1");
            const auto metrics =
                evaluator.evaluation_metrics({strategy_0, strategy_1});
            return py::make_tuple(metrics[0], metrics[1], metrics[2]);
          },
          py::arg("strategy_player_0").noconvert(),
          py::arg("strategy_player_1").noconvert());
}

py::dict openspiel_game(const char *name, const char *gameversion,
                        const char *obstype) {
  py::dict parameters;
  parameters["gameversion"] = gameversion;
  parameters["obstype"] = obstype;

  py::dict game;
  game["spiel_name"] = name;
  game["parameters"] = std::move(parameters);
  return game;
}

py::dict dark_hex_game(const char *gameversion) {
  py::dict game = openspiel_game("dark_hex", gameversion, "reveal-nothing");
  auto parameters = game["parameters"].cast<py::dict>();
  parameters["board_size"] = 3;
  parameters["num_rows"] = 3;
  parameters["num_cols"] = 3;
  return game;
}

py::dict unavailable_openspiel_game(const char *reason) {
  py::dict game;
  game["spiel_name"] = py::none();
  game["parameters"] = py::dict();
  game["reason"] = reason;
  return game;
}
} // namespace

PYBIND11_MODULE(_exp_a_spiel, module) {
  register_liars_dice(module);
  register_evaluator<PtttState<false>>(
      module, "PtttEvaluator",
      openspiel_game("phantom_ttt", "classical", "reveal-nothing"));
  register_evaluator<PtttState<true>>(
      module, "AbruptPtttEvaluator",
      openspiel_game("phantom_ttt", "abrupt", "reveal-nothing"));
  register_evaluator<DhState<false>>(module, "DhEvaluator",
                                     dark_hex_game("cdh"));
  register_evaluator<DhState<true>>(module, "AbruptDhEvaluator",
                                    dark_hex_game("adh"));
  register_evaluator<CornerDhState>(
      module, "CornerDhEvaluator",
      unavailable_openspiel_game(
          "the native corner Dark Hex traverser starts with two occupied "
          "corners, which stock OpenSpiel dark_hex cannot represent"));
}
