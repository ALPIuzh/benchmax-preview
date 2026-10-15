#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <valarray>
#include <vector>

#include <boost/unordered/unordered_flat_map.hpp>
#include "utils.h"

struct InfosetMetadata {
  uint32_t legal_actions;
  uint32_t infoset_id;

  bool operator==(const InfosetMetadata &other) const {
    return legal_actions == other.legal_actions &&
           infoset_id == other.infoset_id;
  }
};

// Maps from infoset to legal action mask
using InfosetMap = boost::unordered_flat_map<uint64_t, InfosetMetadata>;

struct Treeplex {
  InfosetMap infosets;
  std::vector<uint64_t> infoset_keys;
  std::vector<uint32_t> legal_actions;
  std::vector<uint32_t> parent_index;

  uint32_t num_infosets() const { return infoset_keys.size(); }
  bool is_valid_vector(ConstRealBuf buf) const;
  bool is_valid_strategy(ConstRealBuf buf) const;
  void set_uniform(RealBuf buf) const;
  void bh_to_sf(RealBuf buf) const;
  void sf_to_bh(RealBuf buf) const;
  Real br(RealBuf grad, RealBuf strat = std::span<Real>()) const;
  void regret_to_bh(RealBuf buf) const;
};

template <typename T> struct Traverser {
  PerPlayer<std::shared_ptr<Treeplex>> treeplex;
  PerPlayer<std::valarray<Real>> gradients;
  Traverser();

  void compute_gradients(const PerPlayer<ConstRealBuf> strategies);
  std::array<Real, 3>
  evaluation_metrics(const PerPlayer<ConstRealBuf> strategies);
  void compute_openspiel_infostate(const uint8_t player, int64_t i, std::span<bool> buf) const;

private:
  PerPlayer<std::array<std::valarray<Real>, 9>> bufs_;
  PerPlayer<std::valarray<Real>> sf_strategies_;

  void compute_sf_strategies_(const PerPlayer<ConstRealBuf> strategies);
};
