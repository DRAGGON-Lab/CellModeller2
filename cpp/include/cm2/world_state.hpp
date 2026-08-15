#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cm2/types.hpp"

namespace cm2 {

struct CellInit {
  Vec3 position{};
  Vec3 direction{1.0F, 0.0F, 0.0F};
  float length{3.5F};
  float radius{0.5F};
  float growth_rate{1.0F};
  std::int32_t cell_type{0};
};

struct CellSnapshot {
  CellId id{invalid_cell_id};
  Slot slot{invalid_slot};
  Vec3 position{};
  Vec3 direction{1.0F, 0.0F, 0.0F};
  float length{0.0F};
  float radius{0.0F};
  float growth_rate{0.0F};
  std::int32_t cell_type{0};
};

class WorldState {
 public:
  explicit WorldState(std::size_t reserved_capacity = 0);

  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] bool contains(CellId id) const noexcept;

  CellId add_cell(const CellInit& cell);
  std::pair<CellId, CellId> divide_equal(CellId parent_id);
  void advance_growth(float dt);

  [[nodiscard]] CellSnapshot cell(CellId id) const;
  [[nodiscard]] std::vector<CellSnapshot> cells() const;
  [[nodiscard]] std::optional<CellId> lineage_parent(CellId id) const noexcept;

  void validate() const;

 private:
  [[nodiscard]] CellId allocate_id();
  [[nodiscard]] Slot slot_for(CellId id) const;
  void append(CellId id, const CellInit& cell);
  void replace(Slot slot, CellId id, const CellInit& cell);

  CellId next_id_{1};
  std::vector<CellId> ids_;
  std::vector<float> position_x_;
  std::vector<float> position_y_;
  std::vector<float> position_z_;
  std::vector<float> direction_x_;
  std::vector<float> direction_y_;
  std::vector<float> direction_z_;
  std::vector<float> length_;
  std::vector<float> radius_;
  std::vector<float> growth_rate_;
  std::vector<std::int32_t> cell_type_;
  std::unordered_map<CellId, Slot> id_to_slot_;
  std::unordered_map<CellId, CellId> lineage_;
};

}  // namespace cm2
