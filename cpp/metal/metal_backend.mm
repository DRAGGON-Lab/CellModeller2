#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <bit>
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

namespace cm2 {
namespace {

struct alignas(16) MetalFloat4 {
  float x;
  float y;
  float z;
  float w;
};

static_assert(sizeof(MetalFloat4) == 16);

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

class MetalBackend final : public ComputeBackend {
 public:
  MetalBackend() {
    @autoreleasepool {
      device_ = MTLCreateSystemDefaultDevice();
      if (device_ == nil) {
        throw std::runtime_error("Metal is unavailable on this system");
      }
      queue_ = [device_ newCommandQueue];
      if (queue_ == nil) {
        throw std::runtime_error("failed to create a Metal command queue");
      }

      const auto growth_library =
          compile_library(device_, metal::growth_source, "failed to compile Metal growth");
      growth_pipeline_ = compile_pipeline(device_, growth_library, @"advance_growth",
                                          "failed to create the Metal growth pipeline");

      const auto contacts_library =
          compile_library(device_, metal::contacts_source, "failed to compile Metal contacts");
      contact_count_pipeline_ =
          compile_pipeline(device_, contacts_library, @"count_cell_contacts",
                           "failed to create the Metal contact-count pipeline");
      contact_scan_pipeline_ = compile_pipeline(device_, contacts_library, @"inclusive_scan_step",
                                                "failed to create the Metal contact-scan pipeline");
      contact_fill_pipeline_ = compile_pipeline(device_, contacts_library, @"fill_cell_contacts",
                                                "failed to create the Metal contact-fill pipeline");
    }
  }

  [[nodiscard]] BackendInfo info() const override {
    @autoreleasepool {
      const char* device_name = device_.name.UTF8String;
      return {
          .kind = BackendKind::metal,
          .name = "metal",
          .device = device_name == nullptr ? "unknown Apple GPU" : device_name,
          .native = true,
      };
    }
  }

  [[nodiscard]] bool supports(BackendFeature) const noexcept override { return true; }

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

      id<MTLBuffer> scan_input = contact_counts_;
      id<MTLBuffer> scan_output = contact_scan_a_;
      std::uint32_t offset = 1;
      while (offset < pair_slot_count) {
        [encoder setComputePipelineState:contact_scan_pipeline_];
        [encoder setBuffer:scan_input offset:0 atIndex:0];
        [encoder setBuffer:scan_output offset:0 atIndex:1];
        [encoder setBytes:&offset length:sizeof(offset) atIndex:2];
        [encoder setBytes:&pair_slot_count length:sizeof(pair_slot_count) atIndex:3];
        const auto width =
            std::min<NSUInteger>(contact_scan_pipeline_.maxTotalThreadsPerThreadgroup, 256);
        [encoder dispatchThreads:MTLSizeMake(pair_slot_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        scan_input = scan_output;
        scan_output = scan_output == contact_scan_a_ ? contact_scan_b_ : contact_scan_a_;
        if (offset > pair_slot_count / 2) {
          break;
        }
        offset *= 2;
      }
      contact_inclusive_counts_ = scan_input;

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

  id<MTLDevice> device_{nil};
  id<MTLCommandQueue> queue_{nil};
  id<MTLComputePipelineState> growth_pipeline_{nil};
  id<MTLComputePipelineState> contact_count_pipeline_{nil};
  id<MTLComputePipelineState> contact_scan_pipeline_{nil};
  id<MTLComputePipelineState> contact_fill_pipeline_{nil};

  id<MTLBuffer> lengths_{nil};
  id<MTLBuffer> growth_rates_{nil};
  std::size_t growth_capacity_{0};

  id<MTLBuffer> contact_ids_{nil};
  id<MTLBuffer> contact_centers_{nil};
  id<MTLBuffer> contact_axes_{nil};
  id<MTLBuffer> contact_geometry_{nil};
  std::size_t contact_cell_capacity_{0};

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
};

}  // namespace

std::unique_ptr<ComputeBackend> make_metal_backend() { return std::make_unique<MetalBackend>(); }

}  // namespace cm2
