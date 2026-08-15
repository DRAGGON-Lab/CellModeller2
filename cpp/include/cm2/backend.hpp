#pragma once

#include <memory>

#include "cm2/types.hpp"
#include "cm2/world_state.hpp"

namespace cm2 {

class ComputeBackend {
 public:
  virtual ~ComputeBackend() = default;

  [[nodiscard]] virtual BackendInfo info() const = 0;
  virtual void advance_growth(WorldState& state, float dt) = 0;
};

[[nodiscard]] std::unique_ptr<ComputeBackend> make_cpu_backend();
[[nodiscard]] std::unique_ptr<ComputeBackend> make_metal_backend();
[[nodiscard]] bool backend_available(BackendKind kind) noexcept;

}  // namespace cm2
