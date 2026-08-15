#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "cm2/backend.hpp"
#include "cm2/metal/contacts_source.hpp"
#include "cm2/metal/growth_source.hpp"
#include "cm2/metal/mechanics_source.hpp"
#include "cm2/metal/species_source.hpp"

namespace cm2 {
namespace {

struct MetalFloat2 {
  float x;
  float y;
};

static_assert(sizeof(MetalFloat2) == 8);

struct alignas(16) MetalFloat4 {
  float x;
  float y;
  float z;
  float w;
};

static_assert(sizeof(MetalFloat4) == 16);

struct alignas(16) MetalDofs {
  MetalFloat4 linear_length;
  MetalFloat4 rotation;
};

static_assert(sizeof(MetalDofs) == 32);

struct alignas(16) MetalExternalConstraint {
  std::uint64_t id;
  std::uint32_t kind;
  std::uint32_t allowed_region;
  MetalFloat4 geometry;
  MetalFloat4 parameters;
};

static_assert(sizeof(MetalExternalConstraint) == 48);

struct MetalRateInstruction {
  std::uint32_t operation;
  std::uint32_t first;
  std::uint32_t second;
  std::uint32_t third;
  float value;
};

static_assert(sizeof(MetalRateInstruction) == 20);

[[noreturn]] void throw_metal_error(const char* operation, NSError* error) {
  const char* detail = error == nil ? "unknown Metal error" : error.localizedDescription.UTF8String;
  throw std::runtime_error(std::string(operation) + ": " + detail);
}

id<MTLLibrary> compile_library(id<MTLDevice> device, const char* source_text,
                               const char* operation) {
  NSString* source = [NSString stringWithUTF8String:source_text];
  if (source == nil) {
    throw std::runtime_error(std::string(operation) + ": source is not valid UTF-8");
  }
  NSError* error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
  if (library == nil) {
    throw_metal_error(operation, error);
  }
  return library;
}

id<MTLComputePipelineState> compile_pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                             NSString* function_name, const char* operation) {
  id<MTLFunction> function = [library newFunctionWithName:function_name];
  if (function == nil) {
    throw std::runtime_error(std::string(operation) + ": function is missing from the library");
  }
  NSError* error = nil;
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                               error:&error];
  if (pipeline == nil) {
    throw_metal_error(operation, error);
  }
  return pipeline;
}

id<MTLBuffer> allocate_shared_buffer(id<MTLDevice> device, std::size_t byte_count,
                                     const char* description) {
  id<MTLBuffer> buffer = [device newBufferWithLength:byte_count
                                             options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    throw std::runtime_error(std::string("failed to allocate Metal ") + description);
  }
  return buffer;
}

void wait_for_command(id<MTLCommandBuffer> command_buffer, const char* operation) {
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  if (command_buffer.status == MTLCommandBufferStatusError) {
    throw_metal_error(operation, command_buffer.error);
  }
}

void dispatch_1d(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
                 std::uint32_t count) {
  const auto width = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}

id<MTLDevice> select_metal_device(std::uint32_t device_index) {
  NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
  if (devices.count == 0) {
    id<MTLDevice> default_device = MTLCreateSystemDefaultDevice();
    if (device_index == 0 && default_device != nil) {
      return default_device;
    }
    if (default_device == nil) {
      throw std::runtime_error("Metal is unavailable on this system");
    }
  }
  if (static_cast<NSUInteger>(device_index) >= devices.count) {
    throw std::out_of_range("Metal device index is unavailable");
  }
  return devices[device_index];
}

class MetalBackend final : public ComputeBackend {
 public:
  explicit MetalBackend(std::uint32_t device_index) : device_index_(device_index) {
    @autoreleasepool {
      device_ = select_metal_device(device_index_);
      queue_ = [device_ newCommandQueue];
      if (queue_ == nil) {
        throw std::runtime_error("failed to create a Metal command queue");
      }

      const auto growth_library =
          compile_library(device_, metal::growth_source, "failed to compile Metal growth");
      growth_pipeline_ = compile_pipeline(device_, growth_library, @"advance_growth",
                                          "failed to create the Metal growth pipeline");

      const auto species_library =
          compile_library(device_, metal::species_source, "failed to compile Metal species");
      species_pipeline_ = compile_pipeline(device_, species_library, @"advance_species",
                                           "failed to create the Metal species pipeline");

      const auto contacts_library =
          compile_library(device_, metal::contacts_source, "failed to compile Metal contacts");
      contact_count_pipeline_ =
          compile_pipeline(device_, contacts_library, @"count_cell_contacts",
                           "failed to create the Metal contact-count pipeline");
      contact_scan_pipeline_ = compile_pipeline(device_, contacts_library, @"inclusive_scan_step",
                                                "failed to create the Metal contact-scan pipeline");
      contact_fill_pipeline_ = compile_pipeline(device_, contacts_library, @"fill_cell_contacts",
                                                "failed to create the Metal contact-fill pipeline");
      external_contact_count_pipeline_ =
          compile_pipeline(device_, contacts_library, @"count_external_contacts",
                           "failed to create the Metal external-contact-count pipeline");
      external_contact_fill_pipeline_ =
          compile_pipeline(device_, contacts_library, @"fill_external_contacts",
                           "failed to create the Metal external-contact-fill pipeline");

      const auto mechanics_library =
          compile_library(device_, metal::mechanics_source, "failed to compile Metal mechanics");
      mechanics_rows_pipeline_ =
          compile_pipeline(device_, mechanics_library, @"build_mechanics_rows",
                           "failed to create the Metal mechanics-row pipeline");
      mechanics_b_pipeline_ = compile_pipeline(device_, mechanics_library, @"apply_mechanics_b",
                                               "failed to create the Metal mechanics-B pipeline");
      mechanics_transpose_pipeline_ =
          compile_pipeline(device_, mechanics_library, @"apply_mechanics_transpose",
                           "failed to create the Metal mechanics-transpose pipeline");
      mechanics_regularizer_pipeline_ =
          compile_pipeline(device_, mechanics_library, @"add_mechanics_regularizer",
                           "failed to create the Metal mechanics-regularizer pipeline");
      mechanics_initialize_pipeline_ =
          compile_pipeline(device_, mechanics_library, @"initialize_mechanics_vectors",
                           "failed to create the Metal mechanics-initialize pipeline");
      mechanics_update_solution_pipeline_ =
          compile_pipeline(device_, mechanics_library, @"update_mechanics_solution_residual",
                           "failed to create the Metal mechanics-update pipeline");
      mechanics_update_search_pipeline_ =
          compile_pipeline(device_, mechanics_library, @"update_mechanics_search_direction",
                           "failed to create the Metal mechanics-search pipeline");
      mechanics_subtract_pipeline_ =
          compile_pipeline(device_, mechanics_library, @"subtract_mechanics_vectors",
                           "failed to create the Metal mechanics-subtract pipeline");
      mechanics_dot_pipeline_ =
          compile_pipeline(device_, mechanics_library, @"mechanics_dot_terms",
                           "failed to create the Metal mechanics-dot pipeline");
      mechanics_reduce_pipeline_ =
          compile_pipeline(device_, mechanics_library, @"reduce_sum_pairs",
                           "failed to create the Metal mechanics-reduction pipeline");
    }
  }

  [[nodiscard]] BackendInfo info() const override {
    @autoreleasepool {
      const char* device_name = device_.name.UTF8String;
      return {
          .kind = BackendKind::metal,
          .name = "metal",
          .device = device_name == nullptr ? "unknown Apple GPU" : device_name,
          .device_index = device_index_,
          .native = true,
      };
    }
  }

  [[nodiscard]] bool supports(BackendFeature feature) const noexcept override {
    return feature == BackendFeature::growth || feature == BackendFeature::species ||
           feature == BackendFeature::cell_contacts || feature == BackendFeature::cell_mechanics ||
           feature == BackendFeature::external_constraints;
  }

  void advance_growth(WorldState& state, float dt) override {
    auto view = state.growth_state();
    if (view.lengths.empty()) {
      return;
    }
    if (view.lengths.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("Metal growth launch exceeds the uint32 index space");
    }
    ensure_growth_capacity(view.lengths.size());

    const auto byte_count = view.lengths.size_bytes();
    std::memcpy(lengths_.contents, view.lengths.data(), byte_count);
    std::memcpy(growth_rates_.contents, view.growth_rates.data(), byte_count);

    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (command_buffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create a Metal growth command");
      }

      const auto count = static_cast<std::uint32_t>(view.lengths.size());
      [encoder setComputePipelineState:growth_pipeline_];
      [encoder setBuffer:lengths_ offset:0 atIndex:0];
      [encoder setBuffer:growth_rates_ offset:0 atIndex:1];
      [encoder setBytes:&dt length:sizeof(dt) atIndex:2];
      [encoder setBytes:&count length:sizeof(count) atIndex:3];

      const auto width = std::min<NSUInteger>(growth_pipeline_.maxTotalThreadsPerThreadgroup, 256);
      [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      [encoder endEncoding];
      wait_for_command(command_buffer, "Metal growth command failed");
    }

    std::memcpy(view.lengths.data(), lengths_.contents, byte_count);
  }

  void advance_species(WorldState& state, const SpeciesRatePlan& plan,
                       std::span<const float> previous_lengths, float dt) override {
    if (!std::isfinite(dt) || dt < 0.0F) {
      throw std::invalid_argument("species time step must be finite and non-negative");
    }
    state.validate();
    plan.validate();
    if (plan.species_count() != state.species_count()) {
      throw std::invalid_argument("species rate plan and world state species counts disagree");
    }
    if (previous_lengths.size() != state.size()) {
      throw std::invalid_argument("previous cell lengths and world state cell counts disagree");
    }
    if (state.empty() || state.species_count() == 0) {
      return;
    }
    if (state.size() > std::numeric_limits<std::uint32_t>::max() ||
        state.species_count() > std::numeric_limits<std::uint32_t>::max() ||
        plan.instructions().size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("Metal species launch exceeds the uint32 index space");
    }
    if (!std::ranges::all_of(previous_lengths,
                             [](float value) { return std::isfinite(value) && value >= 0.0F; })) {
      throw std::invalid_argument("previous cell lengths must be finite and non-negative");
    }
    if (state.size() > std::numeric_limits<std::size_t>::max() / state.species_count() ||
        state.size() > std::numeric_limits<std::size_t>::max() / plan.instructions().size()) {
      throw std::overflow_error("Metal species buffer size overflow");
    }

    const auto level_count = state.size() * state.species_count();
    const auto workspace_count = state.size() * plan.instructions().size();
    if (level_count > std::numeric_limits<std::uint32_t>::max() ||
        workspace_count > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("Metal flattened species storage exceeds the uint32 index space");
    }
    if (level_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        workspace_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        plan.instructions().size() >
            std::numeric_limits<std::size_t>::max() / sizeof(MetalRateInstruction)) {
      throw std::overflow_error("Metal species allocation size overflow");
    }
    ensure_species_capacity(state.size(), level_count, plan.instructions().size(),
                            state.species_count(), workspace_count);

    const auto geometry = state.geometry_state();
    const auto attributes = state.cell_attributes();
    auto species_state = state.species_state();
    std::memcpy(species_levels_.contents, species_state.levels.data(),
                species_state.levels.size_bytes());
    std::memcpy(species_previous_lengths_.contents, previous_lengths.data(),
                previous_lengths.size_bytes());
    std::memcpy(species_growth_rates_.contents, attributes.growth_rates.data(),
                attributes.growth_rates.size_bytes());
    std::memcpy(species_cell_types_.contents, attributes.cell_types.data(),
                attributes.cell_types.size_bytes());
    auto* centers = static_cast<MetalFloat4*>(species_centers_.contents);
    auto* shapes = static_cast<MetalFloat4*>(species_geometry_.contents);
    for (std::size_t index = 0; index < state.size(); ++index) {
      centers[index] = {geometry.position_x[index], geometry.position_y[index],
                        geometry.position_z[index], 0.0F};
      shapes[index] = {geometry.lengths[index], geometry.radii[index], 0.0F, 0.0F};
    }
    auto* instructions = static_cast<MetalRateInstruction*>(species_instructions_.contents);
    for (std::size_t index = 0; index < plan.instructions().size(); ++index) {
      const auto& instruction = plan.instructions()[index];
      instructions[index] = {
          .operation = static_cast<std::uint32_t>(instruction.operation),
          .first = instruction.first,
          .second = instruction.second,
          .third = instruction.third,
          .value = instruction.value,
      };
    }
    std::memcpy(species_outputs_.contents, plan.outputs().data(), plan.outputs().size_bytes());
    *static_cast<std::uint32_t*>(species_error_.contents) = 0;

    const auto cell_count = static_cast<std::uint32_t>(state.size());
    const auto species_count = static_cast<std::uint32_t>(state.species_count());
    const auto instruction_count = static_cast<std::uint32_t>(plan.instructions().size());
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (command_buffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create a Metal species command");
      }
      [encoder setComputePipelineState:species_pipeline_];
      [encoder setBuffer:species_levels_ offset:0 atIndex:0];
      [encoder setBuffer:species_previous_lengths_ offset:0 atIndex:1];
      [encoder setBuffer:species_centers_ offset:0 atIndex:2];
      [encoder setBuffer:species_geometry_ offset:0 atIndex:3];
      [encoder setBuffer:species_growth_rates_ offset:0 atIndex:4];
      [encoder setBuffer:species_cell_types_ offset:0 atIndex:5];
      [encoder setBuffer:species_instructions_ offset:0 atIndex:6];
      [encoder setBuffer:species_outputs_ offset:0 atIndex:7];
      [encoder setBuffer:species_workspace_ offset:0 atIndex:8];
      [encoder setBuffer:species_error_ offset:0 atIndex:9];
      [encoder setBytes:&dt length:sizeof(dt) atIndex:10];
      [encoder setBytes:&species_count length:sizeof(species_count) atIndex:11];
      [encoder setBytes:&instruction_count length:sizeof(instruction_count) atIndex:12];
      [encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:13];
      dispatch_1d(encoder, species_pipeline_, cell_count);
      [encoder endEncoding];
      wait_for_command(command_buffer, "Metal species command failed");
    }
    if (*static_cast<const std::uint32_t*>(species_error_.contents) != 0) {
      throw std::domain_error("Metal species kernel produced a non-finite value");
    }
    std::memcpy(species_state.levels.data(), species_levels_.contents,
                species_state.levels.size_bytes());
  }

  [[nodiscard]] ContactGraph find_cell_contacts(const WorldState& state,
                                                const ContactParameters& parameters) override {
    validate_contact_parameters(parameters);
    const auto geometry = state.geometry_state();
    if (geometry.size() == 0) {
      return ContactGraph{};
    }
    if (geometry.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("Metal contact launch exceeds the uint32 cell index space");
    }
    if (geometry.size() > std::numeric_limits<std::size_t>::max() / geometry.size()) {
      throw std::overflow_error("Metal exhaustive contact pair count overflow");
    }
    const auto pair_slot_count = geometry.size() * geometry.size();
    if (pair_slot_count > std::numeric_limits<std::uint32_t>::max() / 2) {
      throw std::overflow_error("Metal exhaustive contact staging exceeds the uint32 scan space");
    }

    ensure_contact_cell_capacity(geometry.size());
    ensure_contact_pair_capacity(pair_slot_count);
    upload_contact_cells(geometry);
    const auto contact_count =
        count_contacts(static_cast<std::uint32_t>(geometry.size()),
                       static_cast<std::uint32_t>(pair_slot_count), parameters);
    if (contact_count == 0) {
      return ContactGraph(geometry.size(), {});
    }

    ensure_contact_output_capacity(contact_count);
    fill_contacts(static_cast<std::uint32_t>(geometry.size()), parameters);
    return download_contacts(geometry.size(), contact_count);
  }

  [[nodiscard]] ExternalContactGraph find_external_contacts(
      const WorldState& state, const ConstraintSet& constraints,
      const ConstraintContactParameters& parameters) override {
    validate_constraint_contact_parameters(parameters);
    state.validate();
    const auto geometry = state.geometry_state();
    if (geometry.size() == 0 || constraints.empty()) {
      return ExternalContactGraph(geometry.size(), {});
    }
    if (geometry.size() > std::numeric_limits<std::uint32_t>::max() ||
        constraints.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("Metal external-contact launch exceeds the uint32 index space");
    }
    if (geometry.size() > std::numeric_limits<std::size_t>::max() / constraints.size()) {
      throw std::overflow_error("Metal external-contact pair count overflow");
    }
    const auto pair_count = geometry.size() * constraints.size();
    if (pair_count > std::numeric_limits<std::uint32_t>::max() / 2) {
      throw std::overflow_error("Metal external-contact staging exceeds the uint32 scan space");
    }

    ensure_contact_cell_capacity(geometry.size());
    ensure_external_constraint_capacity(constraints.size());
    ensure_contact_pair_capacity(pair_count);
    upload_contact_cells(geometry);
    upload_external_constraints(constraints);
    const auto contact_count = count_external_contacts(
        static_cast<std::uint32_t>(geometry.size()), static_cast<std::uint32_t>(constraints.size()),
        static_cast<std::uint32_t>(pair_count), parameters);
    if (contact_count == 0) {
      return ExternalContactGraph(geometry.size(), {});
    }

    ensure_contact_output_capacity(contact_count);
    fill_external_contacts(static_cast<std::uint32_t>(geometry.size()),
                           static_cast<std::uint32_t>(constraints.size()), parameters);
    return download_external_contacts(geometry.size(), contact_count);
  }

  [[nodiscard]] MechanicsSolveResult solve_cell_mechanics(
      const WorldState& state, const ContactGraph& contacts,
      const ExternalContactGraph& external_contacts,
      const MechanicsParameters& parameters) override {
    validate_mechanics_parameters(parameters);
    state.validate();
    const auto geometry = state.geometry_state();
    if (contacts.cell_count() != geometry.size()) {
      throw std::invalid_argument("contact graph and world state cell counts disagree");
    }
    if (external_contacts.cell_count() != geometry.size()) {
      throw std::invalid_argument("external contact graph and world state cell counts disagree");
    }
    if (external_contacts.size() > std::numeric_limits<std::size_t>::max() - contacts.size()) {
      throw std::overflow_error("Metal mechanics row count overflow");
    }
    const auto row_count = contacts.size() + external_contacts.size();
    if (geometry.size() > std::numeric_limits<std::uint32_t>::max() ||
        row_count > std::numeric_limits<std::uint32_t>::max() / 2) {
      throw std::overflow_error("Metal mechanics exceeds the uint32 index space");
    }

    MechanicsSolveResult result;
    result.corrections.resize(geometry.size());
    if (geometry.size() == 0 || row_count == 0) {
      return result;
    }

    validate_mechanics_contacts(geometry, contacts);
    validate_external_mechanics_contacts(geometry, external_contacts);
    ensure_contact_cell_capacity(geometry.size());
    ensure_contact_output_capacity(row_count);
    ensure_mechanics_capacity(geometry.size(), row_count);
    upload_contact_cells(geometry);
    upload_mechanics_contacts(contacts, external_contacts);
    upload_mechanics_incidence(contacts, external_contacts);

    const auto cell_count = static_cast<std::uint32_t>(geometry.size());
    const auto contact_count = static_cast<std::uint32_t>(row_count);
    auto residual_squared = initialize_mechanics(cell_count, contact_count);
    result.report.initial_residual_rms =
        std::sqrt(residual_squared / static_cast<float>(cell_count));
    result.report.final_residual_rms = result.report.initial_residual_rms;
    if (!std::isfinite(result.report.initial_residual_rms)) {
      result.report.status = SolverStatus::breakdown;
      result.report.breakdown = SolverBreakdown::non_finite_residual;
      return result;
    }
    if (result.report.initial_residual_rms <= parameters.residual_rms_tolerance) {
      return result;
    }

    result.report.status = SolverStatus::iteration_limit;
    const auto maximum_iterations = mechanics_iteration_limit(parameters, geometry.size());
    for (std::uint32_t iteration = 0; iteration < maximum_iterations; ++iteration) {
      const auto curvature = apply_search_direction(cell_count, contact_count, parameters);
      if (!std::isfinite(curvature)) {
        result.report.status = SolverStatus::breakdown;
        result.report.breakdown = SolverBreakdown::non_finite_curvature;
        break;
      }
      if (curvature <= 0.0F) {
        result.report.status = SolverStatus::breakdown;
        result.report.breakdown = SolverBreakdown::non_positive_curvature;
        break;
      }

      const auto alpha = residual_squared / curvature;
      const auto next_residual_squared = update_solution_residual(cell_count, alpha);
      result.report.iterations = iteration + 1;
      const auto recurrence_rms = std::sqrt(next_residual_squared / static_cast<float>(cell_count));
      if (!std::isfinite(recurrence_rms)) {
        result.report.status = SolverStatus::breakdown;
        result.report.breakdown = SolverBreakdown::non_finite_residual;
        break;
      }

      if (recurrence_rms <= parameters.residual_rms_tolerance) {
        residual_squared = recompute_residual(cell_count, contact_count, parameters);
        const auto recomputed_rms = std::sqrt(residual_squared / static_cast<float>(cell_count));
        if (!std::isfinite(recomputed_rms)) {
          result.report.status = SolverStatus::breakdown;
          result.report.breakdown = SolverBreakdown::non_finite_residual;
          break;
        }
        if (recomputed_rms <= parameters.residual_rms_tolerance) {
          result.report.status = SolverStatus::converged;
          break;
        }
        update_search_direction(cell_count, 0.0F);
        continue;
      }

      const auto beta = next_residual_squared / residual_squared;
      update_search_direction(cell_count, beta);
      residual_squared = next_residual_squared;
    }

    residual_squared = recompute_residual(cell_count, contact_count, parameters);
    result.report.final_residual_rms = std::sqrt(residual_squared / static_cast<float>(cell_count));
    if (!std::isfinite(result.report.final_residual_rms) &&
        result.report.status != SolverStatus::breakdown) {
      result.report.status = SolverStatus::breakdown;
      result.report.breakdown = SolverBreakdown::non_finite_residual;
    }
    result.corrections = download_mechanics_solution(geometry.size());
    return result;
  }

 private:
  void ensure_growth_capacity(std::size_t count) {
    if (count <= growth_capacity_) {
      return;
    }
    growth_capacity_ = std::bit_ceil(count);
    const auto byte_count = growth_capacity_ * sizeof(float);
    lengths_ = allocate_shared_buffer(device_, byte_count, "growth lengths");
    growth_rates_ = allocate_shared_buffer(device_, byte_count, "growth rates");
  }

  void ensure_species_capacity(std::size_t cell_count, std::size_t level_count,
                               std::size_t instruction_count, std::size_t species_count,
                               std::size_t workspace_count) {
    if (cell_count > species_cell_capacity_) {
      species_cell_capacity_ = std::bit_ceil(cell_count);
      species_previous_lengths_ = allocate_shared_buffer(
          device_, species_cell_capacity_ * sizeof(float), "species previous lengths");
      species_centers_ = allocate_shared_buffer(
          device_, species_cell_capacity_ * sizeof(MetalFloat4), "species cell centers");
      species_geometry_ = allocate_shared_buffer(
          device_, species_cell_capacity_ * sizeof(MetalFloat4), "species cell geometry");
      species_growth_rates_ = allocate_shared_buffer(
          device_, species_cell_capacity_ * sizeof(float), "species growth rates");
      species_cell_types_ = allocate_shared_buffer(
          device_, species_cell_capacity_ * sizeof(std::int32_t), "species cell types");
    }
    if (level_count > species_level_capacity_) {
      species_level_capacity_ = std::bit_ceil(level_count);
      species_levels_ = allocate_shared_buffer(device_, species_level_capacity_ * sizeof(float),
                                               "species levels");
    }
    if (instruction_count > species_instruction_capacity_) {
      species_instruction_capacity_ = std::bit_ceil(instruction_count);
      species_instructions_ = allocate_shared_buffer(
          device_, species_instruction_capacity_ * sizeof(MetalRateInstruction),
          "species rate instructions");
    }
    if (species_count > species_output_capacity_) {
      species_output_capacity_ = std::bit_ceil(species_count);
      species_outputs_ = allocate_shared_buffer(
          device_, species_output_capacity_ * sizeof(std::uint32_t), "species rate outputs");
    }
    if (workspace_count > species_workspace_capacity_) {
      species_workspace_capacity_ = std::bit_ceil(workspace_count);
      species_workspace_ = allocate_shared_buffer(
          device_, species_workspace_capacity_ * sizeof(float), "species rate workspace");
    }
    if (species_error_ == nil) {
      species_error_ = allocate_shared_buffer(device_, sizeof(std::uint32_t), "species error flag");
    }
  }

  void ensure_contact_cell_capacity(std::size_t count) {
    if (count <= contact_cell_capacity_) {
      return;
    }
    contact_cell_capacity_ = std::bit_ceil(count);
    contact_ids_ = allocate_shared_buffer(device_, contact_cell_capacity_ * sizeof(std::uint64_t),
                                          "contact cell IDs");
    contact_centers_ = allocate_shared_buffer(device_, contact_cell_capacity_ * sizeof(MetalFloat4),
                                              "contact cell centers");
    contact_axes_ = allocate_shared_buffer(device_, contact_cell_capacity_ * sizeof(MetalFloat4),
                                           "contact cell axes");
    contact_geometry_ = allocate_shared_buffer(
        device_, contact_cell_capacity_ * sizeof(MetalFloat4), "contact cell geometry");
  }

  void ensure_contact_pair_capacity(std::size_t count) {
    if (count <= contact_pair_capacity_) {
      return;
    }
    contact_pair_capacity_ = std::bit_ceil(count);
    const auto byte_count = contact_pair_capacity_ * sizeof(std::uint32_t);
    contact_counts_ = allocate_shared_buffer(device_, byte_count, "contact counts");
    contact_scan_a_ = allocate_shared_buffer(device_, byte_count, "contact scan A");
    contact_scan_b_ = allocate_shared_buffer(device_, byte_count, "contact scan B");
  }

  void ensure_external_constraint_capacity(std::size_t count) {
    if (count <= external_constraint_capacity_) {
      return;
    }
    external_constraint_capacity_ = std::bit_ceil(count);
    external_constraints_ = allocate_shared_buffer(
        device_, external_constraint_capacity_ * sizeof(MetalExternalConstraint),
        "external constraints");
  }

  void ensure_contact_output_capacity(std::size_t count) {
    if (count <= contact_output_capacity_) {
      return;
    }
    contact_output_capacity_ = std::bit_ceil(count);
    const auto id_bytes = contact_output_capacity_ * sizeof(std::uint64_t);
    const auto index_bytes = contact_output_capacity_ * sizeof(std::uint32_t);
    const auto vector_bytes = contact_output_capacity_ * sizeof(MetalFloat4);
    const auto scalar_bytes = contact_output_capacity_ * sizeof(float);
    contact_first_ids_ = allocate_shared_buffer(device_, id_bytes, "contact first IDs");
    contact_second_ids_ = allocate_shared_buffer(device_, id_bytes, "contact second IDs");
    contact_first_slots_ = allocate_shared_buffer(device_, index_bytes, "contact first slots");
    contact_second_slots_ = allocate_shared_buffer(device_, index_bytes, "contact second slots");
    contact_ordinals_ = allocate_shared_buffer(device_, index_bytes, "contact ordinals");
    contact_points_ = allocate_shared_buffer(device_, vector_bytes, "contact points");
    contact_normals_ = allocate_shared_buffer(device_, vector_bytes, "contact normals");
    contact_separations_ = allocate_shared_buffer(device_, scalar_bytes, "contact separations");
    contact_weights_ = allocate_shared_buffer(device_, scalar_bytes, "contact weights");
  }

  void upload_contact_cells(const CellGeometryView& geometry) {
    std::memcpy(contact_ids_.contents, geometry.ids.data(), geometry.ids.size_bytes());
    auto* centers = static_cast<MetalFloat4*>(contact_centers_.contents);
    auto* axes = static_cast<MetalFloat4*>(contact_axes_.contents);
    auto* shapes = static_cast<MetalFloat4*>(contact_geometry_.contents);
    for (std::size_t index = 0; index < geometry.size(); ++index) {
      centers[index] = {
          geometry.position_x[index],
          geometry.position_y[index],
          geometry.position_z[index],
          0.0F,
      };
      axes[index] = {
          geometry.direction_x[index],
          geometry.direction_y[index],
          geometry.direction_z[index],
          0.0F,
      };
      shapes[index] = {geometry.lengths[index], geometry.radii[index], 0.0F, 0.0F};
    }
  }

  void upload_external_constraints(const ConstraintSet& constraints) {
    std::vector<MetalExternalConstraint> values;
    values.reserve(constraints.size());
    for (const auto& plane : constraints.planes()) {
      values.push_back({
          .id = plane.id,
          .kind = static_cast<std::uint32_t>(ExternalConstraintKind::plane),
          .allowed_region = 0,
          .geometry = {plane.point.x, plane.point.y, plane.point.z, 0.0F},
          .parameters = {plane.inward_normal.x, plane.inward_normal.y, plane.inward_normal.z,
                         plane.coefficient},
      });
    }
    for (const auto& sphere : constraints.spheres()) {
      values.push_back({
          .id = sphere.id,
          .kind = static_cast<std::uint32_t>(ExternalConstraintKind::sphere),
          .allowed_region = static_cast<std::uint32_t>(sphere.allowed_region),
          .geometry = {sphere.center.x, sphere.center.y, sphere.center.z, sphere.radius},
          .parameters = {0.0F, 0.0F, 0.0F, sphere.coefficient},
      });
    }
    std::ranges::sort(values, {}, &MetalExternalConstraint::id);
    std::memcpy(external_constraints_.contents, values.data(),
                values.size() * sizeof(MetalExternalConstraint));
  }

  void encode_contact_scan(id<MTLComputeCommandEncoder> encoder, std::uint32_t element_count) {
    id<MTLBuffer> scan_input = contact_counts_;
    id<MTLBuffer> scan_output = contact_scan_a_;
    std::uint32_t offset = 1;
    while (offset < element_count) {
      [encoder setComputePipelineState:contact_scan_pipeline_];
      [encoder setBuffer:scan_input offset:0 atIndex:0];
      [encoder setBuffer:scan_output offset:0 atIndex:1];
      [encoder setBytes:&offset length:sizeof(offset) atIndex:2];
      [encoder setBytes:&element_count length:sizeof(element_count) atIndex:3];
      dispatch_1d(encoder, contact_scan_pipeline_, element_count);
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      scan_input = scan_output;
      scan_output = scan_output == contact_scan_a_ ? contact_scan_b_ : contact_scan_a_;
      if (offset > element_count / 2) {
        break;
      }
      offset *= 2;
    }
    contact_inclusive_counts_ = scan_input;
  }

  [[nodiscard]] std::uint32_t count_contacts(std::uint32_t cell_count,
                                             std::uint32_t pair_slot_count,
                                             const ContactParameters& parameters) {
    const MetalFloat4 gpu_parameters{
        parameters.activation_margin,
        parameters.parallel_sine_threshold,
        parameters.degeneracy_epsilon,
        0.0F,
    };

    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (command_buffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create a Metal contact-count command");
      }

      [encoder setComputePipelineState:contact_count_pipeline_];
      [encoder setBuffer:contact_ids_ offset:0 atIndex:0];
      [encoder setBuffer:contact_centers_ offset:0 atIndex:1];
      [encoder setBuffer:contact_axes_ offset:0 atIndex:2];
      [encoder setBuffer:contact_geometry_ offset:0 atIndex:3];
      [encoder setBuffer:contact_counts_ offset:0 atIndex:4];
      [encoder setBytes:&gpu_parameters length:sizeof(gpu_parameters) atIndex:5];
      [encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:6];
      [encoder dispatchThreads:MTLSizeMake(cell_count, cell_count, 1)
          threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

      encode_contact_scan(encoder, pair_slot_count);

      [encoder endEncoding];
      wait_for_command(command_buffer, "Metal contact count or scan failed");
    }

    const auto* inclusive = static_cast<const std::uint32_t*>(contact_inclusive_counts_.contents);
    return inclusive[pair_slot_count - 1];
  }

  void fill_contacts(std::uint32_t cell_count, const ContactParameters& parameters) {
    const MetalFloat4 gpu_parameters{
        parameters.activation_margin,
        parameters.parallel_sine_threshold,
        parameters.degeneracy_epsilon,
        0.0F,
    };

    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (command_buffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create a Metal contact-fill command");
      }

      [encoder setComputePipelineState:contact_fill_pipeline_];
      [encoder setBuffer:contact_ids_ offset:0 atIndex:0];
      [encoder setBuffer:contact_centers_ offset:0 atIndex:1];
      [encoder setBuffer:contact_axes_ offset:0 atIndex:2];
      [encoder setBuffer:contact_geometry_ offset:0 atIndex:3];
      [encoder setBuffer:contact_counts_ offset:0 atIndex:4];
      [encoder setBuffer:contact_inclusive_counts_ offset:0 atIndex:5];
      [encoder setBuffer:contact_first_ids_ offset:0 atIndex:6];
      [encoder setBuffer:contact_second_ids_ offset:0 atIndex:7];
      [encoder setBuffer:contact_first_slots_ offset:0 atIndex:8];
      [encoder setBuffer:contact_second_slots_ offset:0 atIndex:9];
      [encoder setBuffer:contact_ordinals_ offset:0 atIndex:10];
      [encoder setBuffer:contact_points_ offset:0 atIndex:11];
      [encoder setBuffer:contact_normals_ offset:0 atIndex:12];
      [encoder setBuffer:contact_separations_ offset:0 atIndex:13];
      [encoder setBuffer:contact_weights_ offset:0 atIndex:14];
      [encoder setBytes:&gpu_parameters length:sizeof(gpu_parameters) atIndex:15];
      [encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:16];
      [encoder dispatchThreads:MTLSizeMake(cell_count, cell_count, 1)
          threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
      [encoder endEncoding];
      wait_for_command(command_buffer, "Metal contact fill failed");
    }
  }

  [[nodiscard]] ContactGraph download_contacts(std::size_t cell_count,
                                               std::uint32_t contact_count) const {
    const auto* first_ids = static_cast<const std::uint64_t*>(contact_first_ids_.contents);
    const auto* second_ids = static_cast<const std::uint64_t*>(contact_second_ids_.contents);
    const auto* first_slots = static_cast<const std::uint32_t*>(contact_first_slots_.contents);
    const auto* second_slots = static_cast<const std::uint32_t*>(contact_second_slots_.contents);
    const auto* ordinals = static_cast<const std::uint32_t*>(contact_ordinals_.contents);
    const auto* points = static_cast<const MetalFloat4*>(contact_points_.contents);
    const auto* normals = static_cast<const MetalFloat4*>(contact_normals_.contents);
    const auto* separations = static_cast<const float*>(contact_separations_.contents);
    const auto* weights = static_cast<const float*>(contact_weights_.contents);

    std::vector<CellContact> contacts;
    contacts.reserve(contact_count);
    for (std::uint32_t index = 0; index < contact_count; ++index) {
      if (ordinals[index] > 1) {
        throw std::runtime_error("Metal contact kernel produced an invalid ordinal");
      }
      contacts.push_back({
          .first_id = first_ids[index],
          .second_id = second_ids[index],
          .first_slot = first_slots[index],
          .second_slot = second_slots[index],
          .ordinal = static_cast<std::uint8_t>(ordinals[index]),
          .point_on_first = {points[index].x, points[index].y, points[index].z},
          .normal = {normals[index].x, normals[index].y, normals[index].z},
          .signed_separation = separations[index],
          .weight = weights[index],
      });
    }
    std::ranges::sort(contacts, {}, [](const CellContact& contact) {
      return std::tuple{contact.first_id, contact.second_id, contact.ordinal};
    });
    return ContactGraph(cell_count, std::move(contacts));
  }

  [[nodiscard]] std::uint32_t count_external_contacts(
      std::uint32_t cell_count, std::uint32_t constraint_count, std::uint32_t pair_count,
      const ConstraintContactParameters& parameters) {
    const MetalFloat2 gpu_parameters{parameters.activation_margin, parameters.degeneracy_epsilon};
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (command_buffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create a Metal external-contact-count command");
      }

      [encoder setComputePipelineState:external_contact_count_pipeline_];
      [encoder setBuffer:contact_ids_ offset:0 atIndex:0];
      [encoder setBuffer:contact_centers_ offset:0 atIndex:1];
      [encoder setBuffer:contact_axes_ offset:0 atIndex:2];
      [encoder setBuffer:contact_geometry_ offset:0 atIndex:3];
      [encoder setBuffer:external_constraints_ offset:0 atIndex:4];
      [encoder setBuffer:contact_counts_ offset:0 atIndex:5];
      [encoder setBytes:&gpu_parameters length:sizeof(gpu_parameters) atIndex:6];
      [encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:7];
      [encoder setBytes:&constraint_count length:sizeof(constraint_count) atIndex:8];
      [encoder dispatchThreads:MTLSizeMake(constraint_count, cell_count, 1)
          threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      encode_contact_scan(encoder, pair_count);

      [encoder endEncoding];
      wait_for_command(command_buffer, "Metal external contact count or scan failed");
    }

    const auto* inclusive = static_cast<const std::uint32_t*>(contact_inclusive_counts_.contents);
    return inclusive[pair_count - 1];
  }

  void fill_external_contacts(std::uint32_t cell_count, std::uint32_t constraint_count,
                              const ConstraintContactParameters& parameters) {
    const MetalFloat2 gpu_parameters{parameters.activation_margin, parameters.degeneracy_epsilon};
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (command_buffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create a Metal external-contact-fill command");
      }

      [encoder setComputePipelineState:external_contact_fill_pipeline_];
      [encoder setBuffer:contact_ids_ offset:0 atIndex:0];
      [encoder setBuffer:contact_centers_ offset:0 atIndex:1];
      [encoder setBuffer:contact_axes_ offset:0 atIndex:2];
      [encoder setBuffer:contact_geometry_ offset:0 atIndex:3];
      [encoder setBuffer:external_constraints_ offset:0 atIndex:4];
      [encoder setBuffer:contact_counts_ offset:0 atIndex:5];
      [encoder setBuffer:contact_inclusive_counts_ offset:0 atIndex:6];
      [encoder setBuffer:contact_first_ids_ offset:0 atIndex:7];
      [encoder setBuffer:contact_second_ids_ offset:0 atIndex:8];
      [encoder setBuffer:contact_first_slots_ offset:0 atIndex:9];
      [encoder setBuffer:contact_second_slots_ offset:0 atIndex:10];
      [encoder setBuffer:contact_ordinals_ offset:0 atIndex:11];
      [encoder setBuffer:contact_points_ offset:0 atIndex:12];
      [encoder setBuffer:contact_normals_ offset:0 atIndex:13];
      [encoder setBuffer:contact_separations_ offset:0 atIndex:14];
      [encoder setBuffer:contact_weights_ offset:0 atIndex:15];
      [encoder setBytes:&gpu_parameters length:sizeof(gpu_parameters) atIndex:16];
      [encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:17];
      [encoder setBytes:&constraint_count length:sizeof(constraint_count) atIndex:18];
      [encoder dispatchThreads:MTLSizeMake(constraint_count, cell_count, 1)
          threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
      [encoder endEncoding];
      wait_for_command(command_buffer, "Metal external contact fill failed");
    }
  }

  [[nodiscard]] ExternalContactGraph download_external_contacts(std::size_t cell_count,
                                                                std::uint32_t contact_count) const {
    const auto* cell_ids = static_cast<const std::uint64_t*>(contact_first_ids_.contents);
    const auto* constraint_ids = static_cast<const std::uint64_t*>(contact_second_ids_.contents);
    const auto* cell_slots = static_cast<const std::uint32_t*>(contact_first_slots_.contents);
    const auto* constraint_kinds =
        static_cast<const std::uint32_t*>(contact_second_slots_.contents);
    const auto* endpoints = static_cast<const std::uint32_t*>(contact_ordinals_.contents);
    const auto* points = static_cast<const MetalFloat4*>(contact_points_.contents);
    const auto* normals = static_cast<const MetalFloat4*>(contact_normals_.contents);
    const auto* separations = static_cast<const float*>(contact_separations_.contents);
    const auto* weights = static_cast<const float*>(contact_weights_.contents);

    std::vector<ExternalContact> contacts;
    contacts.reserve(contact_count);
    for (std::uint32_t index = 0; index < contact_count; ++index) {
      if (constraint_kinds[index] > static_cast<std::uint32_t>(ExternalConstraintKind::sphere) ||
          endpoints[index] > static_cast<std::uint32_t>(RodEndpoint::positive)) {
        throw std::runtime_error("Metal external-contact kernel produced an invalid tag");
      }
      contacts.push_back({
          .cell_id = cell_ids[index],
          .cell_slot = cell_slots[index],
          .constraint_id = constraint_ids[index],
          .constraint_kind = static_cast<ExternalConstraintKind>(constraint_kinds[index]),
          .endpoint = static_cast<RodEndpoint>(endpoints[index]),
          .point_on_cell = {points[index].x, points[index].y, points[index].z},
          .normal = {normals[index].x, normals[index].y, normals[index].z},
          .signed_separation = separations[index],
          .weight = weights[index],
      });
    }
    std::ranges::sort(contacts, {}, [](const ExternalContact& contact) {
      return std::tuple{contact.cell_id, contact.constraint_id, contact.endpoint};
    });
    return ExternalContactGraph(cell_count, std::move(contacts));
  }

  static void validate_mechanics_contacts(const CellGeometryView& geometry,
                                          const ContactGraph& contacts) {
    for (const auto& contact : contacts.contacts()) {
      const auto first = static_cast<std::size_t>(contact.first_slot);
      const auto second = static_cast<std::size_t>(contact.second_slot);
      if (geometry.ids[first] != contact.first_id || geometry.ids[second] != contact.second_id) {
        throw std::invalid_argument("contact graph identifiers do not match current state slots");
      }
    }
  }

  static void validate_external_mechanics_contacts(const CellGeometryView& geometry,
                                                   const ExternalContactGraph& contacts) {
    for (const auto& contact : contacts.contacts()) {
      const auto cell = static_cast<std::size_t>(contact.cell_slot);
      if (geometry.ids[cell] != contact.cell_id) {
        throw std::invalid_argument(
            "external contact graph identifiers do not match current state slots");
      }
    }
  }

  static std::uint32_t mechanics_iteration_limit(const MechanicsParameters& parameters,
                                                 std::size_t cell_count) {
    if (parameters.max_iterations != 0) {
      return parameters.max_iterations;
    }
    constexpr std::size_t degrees_of_freedom = 7;
    if (cell_count > std::numeric_limits<std::uint32_t>::max() / degrees_of_freedom) {
      throw std::overflow_error("default mechanics iteration limit exceeds uint32");
    }
    return static_cast<std::uint32_t>(cell_count * degrees_of_freedom);
  }

  void ensure_mechanics_capacity(std::size_t cell_count, std::size_t contact_count) {
    if (cell_count > mechanics_cell_capacity_) {
      mechanics_cell_capacity_ = std::bit_ceil(cell_count);
      const auto dof_bytes = mechanics_cell_capacity_ * sizeof(MetalDofs);
      const auto scalar_bytes = mechanics_cell_capacity_ * sizeof(float);
      mechanics_incidence_offsets_ =
          allocate_shared_buffer(device_, (mechanics_cell_capacity_ + 1) * sizeof(std::uint32_t),
                                 "mechanics incidence offsets");
      mechanics_solution_ = allocate_shared_buffer(device_, dof_bytes, "mechanics solution");
      mechanics_rhs_ = allocate_shared_buffer(device_, dof_bytes, "mechanics right-hand side");
      mechanics_residual_ = allocate_shared_buffer(device_, dof_bytes, "mechanics residual");
      mechanics_search_ = allocate_shared_buffer(device_, dof_bytes, "mechanics search direction");
      mechanics_applied_ = allocate_shared_buffer(device_, dof_bytes, "mechanics applied vector");
      mechanics_dot_terms_ = allocate_shared_buffer(device_, scalar_bytes, "mechanics dot terms");
      mechanics_reduce_a_ = allocate_shared_buffer(device_, scalar_bytes, "mechanics reduction A");
      mechanics_reduce_b_ = allocate_shared_buffer(device_, scalar_bytes, "mechanics reduction B");
    }
    if (contact_count > mechanics_contact_capacity_) {
      mechanics_contact_capacity_ = std::bit_ceil(contact_count);
      const auto dof_bytes = mechanics_contact_capacity_ * sizeof(MetalDofs);
      mechanics_first_rows_ =
          allocate_shared_buffer(device_, dof_bytes, "mechanics first Jacobian rows");
      mechanics_second_rows_ =
          allocate_shared_buffer(device_, dof_bytes, "mechanics second Jacobian rows");
      mechanics_row_values_ = allocate_shared_buffer(
          device_, mechanics_contact_capacity_ * sizeof(float), "mechanics row values");
      mechanics_row_rhs_ = allocate_shared_buffer(
          device_, mechanics_contact_capacity_ * sizeof(float), "mechanics row right-hand side");
      mechanics_incidence_indices_ =
          allocate_shared_buffer(device_, mechanics_contact_capacity_ * 2 * sizeof(std::uint32_t),
                                 "mechanics incidence indices");
    }
  }

  void upload_mechanics_contacts(const ContactGraph& contacts,
                                 const ExternalContactGraph& external_contacts) {
    auto* first_slots = static_cast<std::uint32_t*>(contact_first_slots_.contents);
    auto* second_slots = static_cast<std::uint32_t*>(contact_second_slots_.contents);
    auto* points = static_cast<MetalFloat4*>(contact_points_.contents);
    auto* normals = static_cast<MetalFloat4*>(contact_normals_.contents);
    auto* separations = static_cast<float*>(contact_separations_.contents);
    auto* weights = static_cast<float*>(contact_weights_.contents);
    for (std::size_t index = 0; index < contacts.size(); ++index) {
      const auto& contact = contacts.contacts()[index];
      first_slots[index] = contact.first_slot;
      second_slots[index] = contact.second_slot;
      points[index] = {contact.point_on_first.x, contact.point_on_first.y, contact.point_on_first.z,
                       0.0F};
      normals[index] = {contact.normal.x, contact.normal.y, contact.normal.z, 0.0F};
      separations[index] = contact.signed_separation;
      weights[index] = contact.weight;
    }
    for (std::size_t index = 0; index < external_contacts.size(); ++index) {
      const auto output_index = contacts.size() + index;
      const auto& contact = external_contacts.contacts()[index];
      first_slots[output_index] = contact.cell_slot;
      second_slots[output_index] = invalid_slot;
      points[output_index] = {contact.point_on_cell.x, contact.point_on_cell.y,
                              contact.point_on_cell.z, 0.0F};
      normals[output_index] = {contact.normal.x, contact.normal.y, contact.normal.z, 0.0F};
      separations[output_index] = contact.signed_separation;
      weights[output_index] = contact.weight;
    }
  }

  void upload_mechanics_incidence(const ContactGraph& contacts,
                                  const ExternalContactGraph& external_contacts) {
    auto* offsets = static_cast<std::uint32_t*>(mechanics_incidence_offsets_.contents);
    auto* indices = static_cast<std::uint32_t*>(mechanics_incidence_indices_.contents);
    std::uint32_t cursor = 0;
    for (std::size_t slot = 0; slot < contacts.cell_count(); ++slot) {
      offsets[slot] = cursor;
      for (const auto contact_index : contacts.incident_contact_indices(static_cast<Slot>(slot))) {
        indices[cursor++] = static_cast<std::uint32_t>(contact_index);
      }
      for (const auto contact_index :
           external_contacts.incident_contact_indices(static_cast<Slot>(slot))) {
        indices[cursor++] = static_cast<std::uint32_t>(contacts.size() + contact_index);
      }
    }
    offsets[contacts.cell_count()] = cursor;
    if (cursor != contacts.size() * 2 + external_contacts.size()) {
      throw std::logic_error("contact incidence size is inconsistent");
    }
  }

  void encode_build_mechanics_rows(id<MTLComputeCommandEncoder> encoder,
                                   std::uint32_t contact_count) {
    [encoder setComputePipelineState:mechanics_rows_pipeline_];
    [encoder setBuffer:contact_centers_ offset:0 atIndex:0];
    [encoder setBuffer:contact_axes_ offset:0 atIndex:1];
    [encoder setBuffer:contact_geometry_ offset:0 atIndex:2];
    [encoder setBuffer:contact_first_slots_ offset:0 atIndex:3];
    [encoder setBuffer:contact_second_slots_ offset:0 atIndex:4];
    [encoder setBuffer:contact_points_ offset:0 atIndex:5];
    [encoder setBuffer:contact_normals_ offset:0 atIndex:6];
    [encoder setBuffer:contact_separations_ offset:0 atIndex:7];
    [encoder setBuffer:contact_weights_ offset:0 atIndex:8];
    [encoder setBuffer:mechanics_first_rows_ offset:0 atIndex:9];
    [encoder setBuffer:mechanics_second_rows_ offset:0 atIndex:10];
    [encoder setBuffer:mechanics_row_rhs_ offset:0 atIndex:11];
    [encoder setBytes:&contact_count length:sizeof(contact_count) atIndex:12];
    dispatch_1d(encoder, mechanics_rows_pipeline_, contact_count);
  }

  void encode_mechanics_transpose(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> row_values,
                                  id<MTLBuffer> output, std::uint32_t cell_count) {
    [encoder setComputePipelineState:mechanics_transpose_pipeline_];
    [encoder setBuffer:mechanics_first_rows_ offset:0 atIndex:0];
    [encoder setBuffer:mechanics_second_rows_ offset:0 atIndex:1];
    [encoder setBuffer:row_values offset:0 atIndex:2];
    [encoder setBuffer:mechanics_incidence_offsets_ offset:0 atIndex:3];
    [encoder setBuffer:mechanics_incidence_indices_ offset:0 atIndex:4];
    [encoder setBuffer:contact_first_slots_ offset:0 atIndex:5];
    [encoder setBuffer:output offset:0 atIndex:6];
    [encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:7];
    dispatch_1d(encoder, mechanics_transpose_pipeline_, cell_count);
  }

  void encode_mechanics_operator(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> input,
                                 id<MTLBuffer> output, std::uint32_t cell_count,
                                 std::uint32_t contact_count,
                                 const MechanicsParameters& parameters) {
    [encoder setComputePipelineState:mechanics_b_pipeline_];
    [encoder setBuffer:mechanics_first_rows_ offset:0 atIndex:0];
    [encoder setBuffer:mechanics_second_rows_ offset:0 atIndex:1];
    [encoder setBuffer:contact_first_slots_ offset:0 atIndex:2];
    [encoder setBuffer:contact_second_slots_ offset:0 atIndex:3];
    [encoder setBuffer:input offset:0 atIndex:4];
    [encoder setBuffer:mechanics_row_values_ offset:0 atIndex:5];
    [encoder setBytes:&contact_count length:sizeof(contact_count) atIndex:6];
    dispatch_1d(encoder, mechanics_b_pipeline_, contact_count);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    encode_mechanics_transpose(encoder, mechanics_row_values_, output, cell_count);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    const MetalFloat4 gpu_parameters{parameters.mu_a, parameters.gamma, 0.0F, 0.0F};
    [encoder setComputePipelineState:mechanics_regularizer_pipeline_];
    [encoder setBuffer:contact_axes_ offset:0 atIndex:0];
    [encoder setBuffer:contact_geometry_ offset:0 atIndex:1];
    [encoder setBuffer:input offset:0 atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBytes:&gpu_parameters length:sizeof(gpu_parameters) atIndex:4];
    [encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:5];
    dispatch_1d(encoder, mechanics_regularizer_pipeline_, cell_count);
  }

  id<MTLBuffer> encode_mechanics_dot(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> left,
                                     id<MTLBuffer> right, std::uint32_t cell_count) {
    [encoder setComputePipelineState:mechanics_dot_pipeline_];
    [encoder setBuffer:left offset:0 atIndex:0];
    [encoder setBuffer:right offset:0 atIndex:1];
    [encoder setBuffer:mechanics_dot_terms_ offset:0 atIndex:2];
    [encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:3];
    dispatch_1d(encoder, mechanics_dot_pipeline_, cell_count);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    id<MTLBuffer> input = mechanics_dot_terms_;
    id<MTLBuffer> output = mechanics_reduce_a_;
    auto element_count = cell_count;
    while (element_count > 1) {
      const auto output_count = (element_count + 1) / 2;
      [encoder setComputePipelineState:mechanics_reduce_pipeline_];
      [encoder setBuffer:input offset:0 atIndex:0];
      [encoder setBuffer:output offset:0 atIndex:1];
      [encoder setBytes:&element_count length:sizeof(element_count) atIndex:2];
      dispatch_1d(encoder, mechanics_reduce_pipeline_, output_count);
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      input = output;
      output = output == mechanics_reduce_a_ ? mechanics_reduce_b_ : mechanics_reduce_a_;
      element_count = output_count;
    }
    return input;
  }

  static float read_reduction(id<MTLBuffer> reduction) {
    return *static_cast<const float*>(reduction.contents);
  }

  [[nodiscard]] float initialize_mechanics(std::uint32_t cell_count, std::uint32_t contact_count) {
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (command_buffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create a Metal mechanics initialization command");
      }
      encode_build_mechanics_rows(encoder, contact_count);
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      encode_mechanics_transpose(encoder, mechanics_row_rhs_, mechanics_rhs_, cell_count);
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

      [encoder setComputePipelineState:mechanics_initialize_pipeline_];
      [encoder setBuffer:mechanics_rhs_ offset:0 atIndex:0];
      [encoder setBuffer:mechanics_solution_ offset:0 atIndex:1];
      [encoder setBuffer:mechanics_residual_ offset:0 atIndex:2];
      [encoder setBuffer:mechanics_search_ offset:0 atIndex:3];
      [encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:4];
      dispatch_1d(encoder, mechanics_initialize_pipeline_, cell_count);
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

      const auto reduction =
          encode_mechanics_dot(encoder, mechanics_residual_, mechanics_residual_, cell_count);
      [encoder endEncoding];
      wait_for_command(command_buffer, "Metal mechanics initialization failed");
      return read_reduction(reduction);
    }
  }

  [[nodiscard]] float apply_search_direction(std::uint32_t cell_count, std::uint32_t contact_count,
                                             const MechanicsParameters& parameters) {
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (command_buffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create a Metal mechanics operator command");
      }
      encode_mechanics_operator(encoder, mechanics_search_, mechanics_applied_, cell_count,
                                contact_count, parameters);
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      const auto reduction =
          encode_mechanics_dot(encoder, mechanics_search_, mechanics_applied_, cell_count);
      [encoder endEncoding];
      wait_for_command(command_buffer, "Metal mechanics operator application failed");
      return read_reduction(reduction);
    }
  }

  [[nodiscard]] float update_solution_residual(std::uint32_t cell_count, float alpha) {
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (command_buffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create a Metal mechanics update command");
      }
      [encoder setComputePipelineState:mechanics_update_solution_pipeline_];
      [encoder setBuffer:mechanics_solution_ offset:0 atIndex:0];
      [encoder setBuffer:mechanics_residual_ offset:0 atIndex:1];
      [encoder setBuffer:mechanics_search_ offset:0 atIndex:2];
      [encoder setBuffer:mechanics_applied_ offset:0 atIndex:3];
      [encoder setBytes:&alpha length:sizeof(alpha) atIndex:4];
      [encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:5];
      dispatch_1d(encoder, mechanics_update_solution_pipeline_, cell_count);
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      const auto reduction =
          encode_mechanics_dot(encoder, mechanics_residual_, mechanics_residual_, cell_count);
      [encoder endEncoding];
      wait_for_command(command_buffer, "Metal mechanics update failed");
      return read_reduction(reduction);
    }
  }

  void update_search_direction(std::uint32_t cell_count, float beta) {
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (command_buffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create a Metal mechanics search command");
      }
      [encoder setComputePipelineState:mechanics_update_search_pipeline_];
      [encoder setBuffer:mechanics_residual_ offset:0 atIndex:0];
      [encoder setBuffer:mechanics_search_ offset:0 atIndex:1];
      [encoder setBytes:&beta length:sizeof(beta) atIndex:2];
      [encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:3];
      dispatch_1d(encoder, mechanics_update_search_pipeline_, cell_count);
      [encoder endEncoding];
      wait_for_command(command_buffer, "Metal mechanics search update failed");
    }
  }

  [[nodiscard]] float recompute_residual(std::uint32_t cell_count, std::uint32_t contact_count,
                                         const MechanicsParameters& parameters) {
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (command_buffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create a Metal mechanics residual command");
      }
      encode_mechanics_operator(encoder, mechanics_solution_, mechanics_applied_, cell_count,
                                contact_count, parameters);
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

      [encoder setComputePipelineState:mechanics_subtract_pipeline_];
      [encoder setBuffer:mechanics_rhs_ offset:0 atIndex:0];
      [encoder setBuffer:mechanics_applied_ offset:0 atIndex:1];
      [encoder setBuffer:mechanics_residual_ offset:0 atIndex:2];
      [encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:3];
      dispatch_1d(encoder, mechanics_subtract_pipeline_, cell_count);
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      const auto reduction =
          encode_mechanics_dot(encoder, mechanics_residual_, mechanics_residual_, cell_count);
      [encoder endEncoding];
      wait_for_command(command_buffer, "Metal mechanics residual recomputation failed");
      return read_reduction(reduction);
    }
  }

  [[nodiscard]] std::vector<CellCorrection> download_mechanics_solution(
      std::size_t cell_count) const {
    const auto* values = static_cast<const MetalDofs*>(mechanics_solution_.contents);
    std::vector<CellCorrection> result;
    result.reserve(cell_count);
    for (std::size_t index = 0; index < cell_count; ++index) {
      result.push_back({
          .translation = {values[index].linear_length.x, values[index].linear_length.y,
                          values[index].linear_length.z},
          .rotation = {values[index].rotation.x, values[index].rotation.y,
                       values[index].rotation.z},
          .length = values[index].linear_length.w,
      });
    }
    return result;
  }

  std::uint32_t device_index_{0};
  id<MTLDevice> device_{nil};
  id<MTLCommandQueue> queue_{nil};
  id<MTLComputePipelineState> growth_pipeline_{nil};
  id<MTLComputePipelineState> species_pipeline_{nil};
  id<MTLComputePipelineState> contact_count_pipeline_{nil};
  id<MTLComputePipelineState> contact_scan_pipeline_{nil};
  id<MTLComputePipelineState> contact_fill_pipeline_{nil};
  id<MTLComputePipelineState> external_contact_count_pipeline_{nil};
  id<MTLComputePipelineState> external_contact_fill_pipeline_{nil};
  id<MTLComputePipelineState> mechanics_rows_pipeline_{nil};
  id<MTLComputePipelineState> mechanics_b_pipeline_{nil};
  id<MTLComputePipelineState> mechanics_transpose_pipeline_{nil};
  id<MTLComputePipelineState> mechanics_regularizer_pipeline_{nil};
  id<MTLComputePipelineState> mechanics_initialize_pipeline_{nil};
  id<MTLComputePipelineState> mechanics_update_solution_pipeline_{nil};
  id<MTLComputePipelineState> mechanics_update_search_pipeline_{nil};
  id<MTLComputePipelineState> mechanics_subtract_pipeline_{nil};
  id<MTLComputePipelineState> mechanics_dot_pipeline_{nil};
  id<MTLComputePipelineState> mechanics_reduce_pipeline_{nil};

  id<MTLBuffer> lengths_{nil};
  id<MTLBuffer> growth_rates_{nil};
  std::size_t growth_capacity_{0};

  id<MTLBuffer> species_levels_{nil};
  id<MTLBuffer> species_previous_lengths_{nil};
  id<MTLBuffer> species_centers_{nil};
  id<MTLBuffer> species_geometry_{nil};
  id<MTLBuffer> species_growth_rates_{nil};
  id<MTLBuffer> species_cell_types_{nil};
  id<MTLBuffer> species_instructions_{nil};
  id<MTLBuffer> species_outputs_{nil};
  id<MTLBuffer> species_workspace_{nil};
  id<MTLBuffer> species_error_{nil};
  std::size_t species_cell_capacity_{0};
  std::size_t species_level_capacity_{0};
  std::size_t species_instruction_capacity_{0};
  std::size_t species_output_capacity_{0};
  std::size_t species_workspace_capacity_{0};

  id<MTLBuffer> contact_ids_{nil};
  id<MTLBuffer> contact_centers_{nil};
  id<MTLBuffer> contact_axes_{nil};
  id<MTLBuffer> contact_geometry_{nil};
  std::size_t contact_cell_capacity_{0};

  id<MTLBuffer> external_constraints_{nil};
  std::size_t external_constraint_capacity_{0};

  id<MTLBuffer> contact_counts_{nil};
  id<MTLBuffer> contact_scan_a_{nil};
  id<MTLBuffer> contact_scan_b_{nil};
  id<MTLBuffer> contact_inclusive_counts_{nil};
  std::size_t contact_pair_capacity_{0};

  id<MTLBuffer> contact_first_ids_{nil};
  id<MTLBuffer> contact_second_ids_{nil};
  id<MTLBuffer> contact_first_slots_{nil};
  id<MTLBuffer> contact_second_slots_{nil};
  id<MTLBuffer> contact_ordinals_{nil};
  id<MTLBuffer> contact_points_{nil};
  id<MTLBuffer> contact_normals_{nil};
  id<MTLBuffer> contact_separations_{nil};
  id<MTLBuffer> contact_weights_{nil};
  std::size_t contact_output_capacity_{0};

  id<MTLBuffer> mechanics_first_rows_{nil};
  id<MTLBuffer> mechanics_second_rows_{nil};
  id<MTLBuffer> mechanics_row_values_{nil};
  id<MTLBuffer> mechanics_row_rhs_{nil};
  id<MTLBuffer> mechanics_incidence_offsets_{nil};
  id<MTLBuffer> mechanics_incidence_indices_{nil};
  id<MTLBuffer> mechanics_solution_{nil};
  id<MTLBuffer> mechanics_rhs_{nil};
  id<MTLBuffer> mechanics_residual_{nil};
  id<MTLBuffer> mechanics_search_{nil};
  id<MTLBuffer> mechanics_applied_{nil};
  id<MTLBuffer> mechanics_dot_terms_{nil};
  id<MTLBuffer> mechanics_reduce_a_{nil};
  id<MTLBuffer> mechanics_reduce_b_{nil};
  std::size_t mechanics_cell_capacity_{0};
  std::size_t mechanics_contact_capacity_{0};
};

}  // namespace

std::unique_ptr<ComputeBackend> make_metal_backend(std::uint32_t device_index) {
  return std::make_unique<MetalBackend>(device_index);
}

std::size_t metal_backend_device_count() noexcept {
  @autoreleasepool {
    const auto count = MTLCopyAllDevices().count;
    return count == 0 && MTLCreateSystemDefaultDevice() != nil ? 1 : count;
  }
}

}  // namespace cm2
