#include <cuda_runtime_api.h>

#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "cm2/backend.hpp"
#include "kernels/growth.cuh"

namespace cm2 {
namespace {

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

class CudaBackend final : public ComputeBackend {
 public:
  CudaBackend() {
    check_cuda(cudaGetDevice(&device_index_), "failed to select a CUDA device");
    check_cuda(cudaGetDeviceProperties(&device_properties_, device_index_),
               "failed to inspect the CUDA device");
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "failed to create a CUDA stream");
  }

  ~CudaBackend() override {
    if (lengths_ != nullptr) {
      cudaFree(lengths_);
    }
    if (growth_rates_ != nullptr) {
      cudaFree(growth_rates_);
    }
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  [[nodiscard]] BackendInfo info() const override {
    return {
        .kind = BackendKind::cuda,
        .name = "cuda",
        .device = device_properties_.name,
        .native = true,
    };
  }

  void advance_growth(WorldState& state, float dt) override {
    auto view = state.growth_state();
    if (view.lengths.empty()) {
      return;
    }
    if (view.lengths.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("CUDA growth launch exceeds the uint32 index space");
    }
    ensure_capacity(view.lengths.size());

    const auto byte_count = view.lengths.size_bytes();
    check_cuda(
        cudaMemcpyAsync(lengths_, view.lengths.data(), byte_count, cudaMemcpyHostToDevice, stream_),
        "failed to upload CUDA growth lengths");
    check_cuda(cudaMemcpyAsync(growth_rates_, view.growth_rates.data(), byte_count,
                               cudaMemcpyHostToDevice, stream_),
               "failed to upload CUDA growth rates");

    cuda::launch_growth(lengths_, growth_rates_, dt,
                        static_cast<std::uint32_t>(view.lengths.size()), stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA growth kernel");
    check_cuda(
        cudaMemcpyAsync(view.lengths.data(), lengths_, byte_count, cudaMemcpyDeviceToHost, stream_),
        "failed to download CUDA growth lengths");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA growth execution failed");
  }

  [[nodiscard]] ContactGraph find_cell_contacts(const WorldState&,
                                                const ContactParameters&) override {
    throw std::runtime_error("CUDA cell contacts are not implemented in this build");
  }

 private:
  void ensure_capacity(std::size_t count) {
    if (count <= capacity_) {
      return;
    }

    const auto new_capacity = std::bit_ceil(count);
    const auto byte_count = new_capacity * sizeof(float);
    float* new_lengths = nullptr;
    float* new_growth_rates = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&new_lengths), byte_count),
               "failed to allocate CUDA growth lengths");
    const auto rates_result = cudaMalloc(reinterpret_cast<void**>(&new_growth_rates), byte_count);
    if (rates_result != cudaSuccess) {
      cudaFree(new_lengths);
      check_cuda(rates_result, "failed to allocate CUDA growth rates");
    }

    const auto lengths_result = lengths_ == nullptr ? cudaSuccess : cudaFree(lengths_);
    const auto rates_release_result =
        growth_rates_ == nullptr ? cudaSuccess : cudaFree(growth_rates_);
    lengths_ = nullptr;
    growth_rates_ = nullptr;
    if (lengths_result != cudaSuccess || rates_release_result != cudaSuccess) {
      cudaFree(new_lengths);
      cudaFree(new_growth_rates);
      check_cuda(lengths_result != cudaSuccess ? lengths_result : rates_release_result,
                 "failed to release old CUDA growth buffers");
    }
    lengths_ = new_lengths;
    growth_rates_ = new_growth_rates;
    capacity_ = new_capacity;
  }

  int device_index_{0};
  cudaDeviceProp device_properties_{};
  cudaStream_t stream_{nullptr};
  float* lengths_{nullptr};
  float* growth_rates_{nullptr};
  std::size_t capacity_{0};
};

}  // namespace

std::unique_ptr<ComputeBackend> make_cuda_backend() { return std::make_unique<CudaBackend>(); }

}  // namespace cm2
