#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "cm2/backend.hpp"

namespace cm2 {

class Simulation {
 public:
  explicit Simulation(BackendKind backend = BackendKind::cpu, std::size_t reserved_capacity = 0);

  [[nodiscard]] BackendInfo backend_info() const;
  [[nodiscard]] double time() const noexcept;
  [[nodiscard]] std::size_t cell_count() const noexcept;

  CellId add_cell(const CellInit& cell);
  std::pair<CellId, CellId> divide_equal(CellId parent_id);
  void step(float dt);
  [[nodiscard]] ContactGraph find_cell_contacts(
      const ContactParameters& parameters = ContactParameters{});

  [[nodiscard]] CellSnapshot cell(CellId id) const;
  [[nodiscard]] std::vector<CellSnapshot> cells() const;
  [[nodiscard]] std::optional<CellId> lineage_parent(CellId id) const noexcept;
  void validate() const;

 private:
  WorldState state_;
  std::unique_ptr<ComputeBackend> backend_;
  double time_{0.0};
};

}  // namespace cm2
