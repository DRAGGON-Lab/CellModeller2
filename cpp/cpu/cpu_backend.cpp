#include <memory>

#include "cm2/backend.hpp"

namespace cm2 {
namespace {

class CpuBackend final : public ComputeBackend {
 public:
  [[nodiscard]] BackendInfo info() const override {
    return {
        .kind = BackendKind::cpu,
        .name = "cpu-reference",
        .device = "host",
        .native = true,
    };
  }

  void advance_growth(WorldState& state, float dt) override { state.advance_growth(dt); }
};

}  // namespace

std::unique_ptr<ComputeBackend> make_cpu_backend() { return std::make_unique<CpuBackend>(); }

}  // namespace cm2
