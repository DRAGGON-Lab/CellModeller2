#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "cm2/backend.hpp"
#include "cm2/metal/growth_source.hpp"

namespace cm2 {
namespace {

[[noreturn]] void throw_metal_error(const char* operation, NSError* error) {
  const char* detail = error == nil ? "unknown Metal error" : error.localizedDescription.UTF8String;
  throw std::runtime_error(std::string(operation) + ": " + detail);
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

      NSString* source = [NSString stringWithUTF8String:metal::growth_source];
      if (source == nil) {
        throw std::runtime_error("failed to encode the Metal growth source");
      }
      NSError* error = nil;
      id<MTLLibrary> library = [device_ newLibraryWithSource:source options:nil error:&error];
      if (library == nil) {
        throw_metal_error("failed to compile the Metal growth library", error);
      }
      id<MTLFunction> function = [library newFunctionWithName:@"advance_growth"];
      if (function == nil) {
        throw std::runtime_error("Metal growth library does not contain advance_growth");
      }
      pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
      if (pipeline_ == nil) {
        throw_metal_error("failed to create the Metal growth pipeline", error);
      }
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

  void advance_growth(WorldState& state, float dt) override {
    auto view = state.growth_state();
    if (view.lengths.empty()) {
      return;
    }
    if (view.lengths.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("Metal growth launch exceeds the uint32 index space");
    }
    ensure_capacity(view.lengths.size());

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
      [encoder setComputePipelineState:pipeline_];
      [encoder setBuffer:lengths_ offset:0 atIndex:0];
      [encoder setBuffer:growth_rates_ offset:0 atIndex:1];
      [encoder setBytes:&dt length:sizeof(dt) atIndex:2];
      [encoder setBytes:&count length:sizeof(count) atIndex:3];

      const auto width = std::min<NSUInteger>(pipeline_.maxTotalThreadsPerThreadgroup, 256);
      [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      [encoder endEncoding];
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      if (command_buffer.status == MTLCommandBufferStatusError) {
        throw_metal_error("Metal growth command failed", command_buffer.error);
      }
    }

    std::memcpy(view.lengths.data(), lengths_.contents, byte_count);
  }

 private:
  void ensure_capacity(std::size_t count) {
    if (count <= capacity_) {
      return;
    }
    capacity_ = std::bit_ceil(count);
    const auto byte_count = capacity_ * sizeof(float);
    lengths_ = [device_ newBufferWithLength:byte_count options:MTLResourceStorageModeShared];
    growth_rates_ = [device_ newBufferWithLength:byte_count options:MTLResourceStorageModeShared];
    if (lengths_ == nil || growth_rates_ == nil) {
      throw std::runtime_error("failed to allocate Metal growth buffers");
    }
  }

  id<MTLDevice> device_{nil};
  id<MTLCommandQueue> queue_{nil};
  id<MTLComputePipelineState> pipeline_{nil};
  id<MTLBuffer> lengths_{nil};
  id<MTLBuffer> growth_rates_{nil};
  std::size_t capacity_{0};
};

}  // namespace

std::unique_ptr<ComputeBackend> make_metal_backend() { return std::make_unique<MetalBackend>(); }

}  // namespace cm2
