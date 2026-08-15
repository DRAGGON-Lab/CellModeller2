#include "cm2/contact_graph.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cm2 {
namespace {

bool finite(const Vec3& value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

std::size_t checked_offset_count(std::size_t cell_count) {
  if (cell_count > static_cast<std::size_t>(invalid_slot) ||
      cell_count == std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("contact graph exceeds the slot index space");
  }
  return cell_count + 1;
}

void validate_contact(const CellContact& contact, std::size_t cell_count) {
  if (contact.first_id == invalid_cell_id || contact.second_id == invalid_cell_id ||
      contact.first_id >= contact.second_id) {
    throw std::invalid_argument("contact cell identifiers are not canonical");
  }
  if (contact.first_slot >= cell_count || contact.second_slot >= cell_count ||
      contact.first_slot == contact.second_slot) {
    throw std::invalid_argument("contact slots are invalid");
  }
  if (contact.ordinal > 1) {
    throw std::invalid_argument("cell contact ordinal exceeds the capsule-pair contract");
  }
  if (!finite(contact.point_on_first) || !finite(contact.normal) ||
      !std::isfinite(contact.signed_separation) || !std::isfinite(contact.weight) ||
      contact.weight <= 0.0F) {
    throw std::invalid_argument("contact contains a non-finite or invalid field");
  }
  if (std::abs(norm(contact.normal) - 1.0F) > 1.0e-4F) {
    throw std::invalid_argument("contact normal is not unit length");
  }
}

}  // namespace

ContactGraph::ContactGraph(std::size_t cell_count, std::vector<CellContact> contacts)
    : cell_count_(cell_count),
      contacts_(std::move(contacts)),
      incidence_offsets_(checked_offset_count(cell_count)) {
  if (contacts_.size() > std::numeric_limits<std::size_t>::max() / 2) {
    throw std::overflow_error("contact incidence index size overflow");
  }

  for (const auto& contact : contacts_) {
    validate_contact(contact, cell_count_);
    ++incidence_offsets_[static_cast<std::size_t>(contact.first_slot) + 1];
    ++incidence_offsets_[static_cast<std::size_t>(contact.second_slot) + 1];
  }
  for (std::size_t index = 1; index < incidence_offsets_.size(); ++index) {
    incidence_offsets_[index] += incidence_offsets_[index - 1];
  }

  incidence_contact_indices_.resize(contacts_.size() * 2);
  auto cursors = incidence_offsets_;
  for (std::size_t index = 0; index < contacts_.size(); ++index) {
    const auto& contact = contacts_[index];
    incidence_contact_indices_[cursors[contact.first_slot]++] = index;
    incidence_contact_indices_[cursors[contact.second_slot]++] = index;
  }
}

std::size_t ContactGraph::cell_count() const noexcept { return cell_count_; }

std::size_t ContactGraph::size() const noexcept { return contacts_.size(); }

bool ContactGraph::empty() const noexcept { return contacts_.empty(); }

std::span<const CellContact> ContactGraph::contacts() const& noexcept { return contacts_; }

std::span<const std::size_t> ContactGraph::incident_contact_indices(Slot slot) const& {
  const auto index = static_cast<std::size_t>(slot);
  if (index >= cell_count_) {
    throw std::out_of_range("contact incidence slot is out of range");
  }
  const auto begin = incidence_offsets_[index];
  const auto end = incidence_offsets_[index + 1];
  return std::span<const std::size_t>(incidence_contact_indices_).subspan(begin, end - begin);
}

}  // namespace cm2
