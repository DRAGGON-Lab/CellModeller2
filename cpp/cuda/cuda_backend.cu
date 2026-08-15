#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "cm2/backend.hpp"
#include "kernels/contacts.cuh"
#include "kernels/growth.cuh"

namespace cm2 {
namespace {

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

template <typename T>
class CudaBuffer {
 public:
  CudaBuffer() = default;
  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;

  ~CudaBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  void reserve(std::size_t count, const char* description) {
    if (count <= capacity_) {
      return;
    }
    const auto new_capacity = std::bit_ceil(count);
    if (new_capacity > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::overflow_error(std::string("CUDA buffer size overflow for ") + description);
    }

    T* replacement = nullptr;
    const auto byte_count = new_capacity * sizeof(T);
    const auto allocation_operation = std::string("failed to allocate CUDA ") + description;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&replacement), byte_count),
               allocation_operation.c_str());
    if (data_ != nullptr) {
      const auto release_result = cudaFree(data_);
      if (release_result != cudaSuccess) {
        cudaFree(replacement);
        const auto release_operation = std::string("failed to release old CUDA ") + description;
        check_cuda(release_result, release_operation.c_str());
      }
    }
    data_ = replacement;
    capacity_ = new_capacity;
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }

 private:
  T* data_{nullptr};
  std::size_t capacity_{0};
};

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

  [[nodiscard]] bool supports(BackendFeature feature) const noexcept override {
    return feature == BackendFeature::growth || feature == BackendFeature::cell_contacts;
  }

  void advance_growth(WorldState& state, float dt) override {
    auto view = state.growth_state();
    if (view.lengths.empty()) {
      return;
    }
    if (view.lengths.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("CUDA growth launch exceeds the uint32 index space");
    }
    lengths_.reserve(view.lengths.size(), "growth lengths");
    growth_rates_.reserve(view.growth_rates.size(), "growth rates");

    const auto byte_count = view.lengths.size_bytes();
    check_cuda(cudaMemcpyAsync(lengths_.data(), view.lengths.data(), byte_count,
                               cudaMemcpyHostToDevice, stream_),
               "failed to upload CUDA growth lengths");
    check_cuda(cudaMemcpyAsync(growth_rates_.data(), view.growth_rates.data(), byte_count,
                               cudaMemcpyHostToDevice, stream_),
               "failed to upload CUDA growth rates");

    cuda::launch_growth(lengths_.data(), growth_rates_.data(), dt,
                        static_cast<std::uint32_t>(view.lengths.size()), stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA growth kernel");
    check_cuda(cudaMemcpyAsync(view.lengths.data(), lengths_.data(), byte_count,
                               cudaMemcpyDeviceToHost, stream_),
               "failed to download CUDA growth lengths");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA growth execution failed");
  }

  [[nodiscard]] ContactGraph find_cell_contacts(const WorldState& state,
                                                const ContactParameters& parameters) override {
    validate_contact_parameters(parameters);
    const auto geometry = state.geometry_state();
    if (geometry.size() == 0) {
      return ContactGraph{};
    }
    if (geometry.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("CUDA contact launch exceeds the uint32 cell index space");
    }
    if (geometry.size() > std::numeric_limits<std::size_t>::max() / geometry.size()) {
      throw std::overflow_error("CUDA exhaustive contact pair count overflow");
    }
    const auto pair_slot_count = geometry.size() * geometry.size();
    if (pair_slot_count > std::numeric_limits<std::uint32_t>::max() / 2) {
      throw std::overflow_error("CUDA exhaustive contact staging exceeds the uint32 scan space");
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
  void ensure_contact_cell_capacity(std::size_t count) {
    contact_ids_.reserve(count, "contact cell IDs");
    contact_centers_.reserve(count, "contact cell centers");
    contact_axes_.reserve(count, "contact cell axes");
    contact_geometry_.reserve(count, "contact cell geometry");
  }

  void ensure_contact_pair_capacity(std::size_t count) {
    contact_counts_.reserve(count, "contact counts");
    contact_scan_a_.reserve(count, "contact scan A");
    contact_scan_b_.reserve(count, "contact scan B");
  }

  void ensure_contact_output_capacity(std::size_t count) {
    contact_first_ids_.reserve(count, "contact first IDs");
    contact_second_ids_.reserve(count, "contact second IDs");
    contact_first_slots_.reserve(count, "contact first slots");
    contact_second_slots_.reserve(count, "contact second slots");
    contact_ordinals_.reserve(count, "contact ordinals");
    contact_points_.reserve(count, "contact points");
    contact_normals_.reserve(count, "contact normals");
    contact_separations_.reserve(count, "contact separations");
    contact_weights_.reserve(count, "contact weights");
  }

  void upload_contact_cells(const CellGeometryView& geometry) {
    std::vector<float4> centers(geometry.size());
    std::vector<float4> axes(geometry.size());
    std::vector<float4> shapes(geometry.size());
    for (std::size_t index = 0; index < geometry.size(); ++index) {
      centers[index] = make_float4(geometry.position_x[index], geometry.position_y[index],
                                   geometry.position_z[index], 0.0F);
      axes[index] = make_float4(geometry.direction_x[index], geometry.direction_y[index],
                                geometry.direction_z[index], 0.0F);
      shapes[index] = make_float4(geometry.lengths[index], geometry.radii[index], 0.0F, 0.0F);
    }

    check_cuda(cudaMemcpy(contact_ids_.data(), geometry.ids.data(), geometry.ids.size_bytes(),
                          cudaMemcpyHostToDevice),
               "failed to upload CUDA contact cell IDs");
    check_cuda(cudaMemcpy(contact_centers_.data(), centers.data(), centers.size() * sizeof(float4),
                          cudaMemcpyHostToDevice),
               "failed to upload CUDA contact cell centers");
    check_cuda(cudaMemcpy(contact_axes_.data(), axes.data(), axes.size() * sizeof(float4),
                          cudaMemcpyHostToDevice),
               "failed to upload CUDA contact cell axes");
    check_cuda(cudaMemcpy(contact_geometry_.data(), shapes.data(), shapes.size() * sizeof(float4),
                          cudaMemcpyHostToDevice),
               "failed to upload CUDA contact cell geometry");
  }

  [[nodiscard]] std::uint32_t count_contacts(std::uint32_t cell_count,
                                             std::uint32_t pair_slot_count,
                                             const ContactParameters& parameters) {
    const cuda::ContactParametersGpu gpu_parameters{
        .activation_margin = parameters.activation_margin,
        .parallel_sine_threshold = parameters.parallel_sine_threshold,
        .degeneracy_epsilon = parameters.degeneracy_epsilon,
    };
    cuda::launch_contact_count(contact_ids_.data(), contact_centers_.data(), contact_axes_.data(),
                               contact_geometry_.data(), contact_counts_.data(), gpu_parameters,
                               cell_count, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA contact-count kernel");

    const std::uint32_t* scan_input = contact_counts_.data();
    std::uint32_t* scan_output = contact_scan_a_.data();
    std::uint32_t offset = 1;
    while (offset < pair_slot_count) {
      cuda::launch_inclusive_scan_step(scan_input, scan_output, offset, pair_slot_count, stream_);
      check_cuda(cudaGetLastError(), "failed to launch the CUDA contact-scan kernel");
      scan_input = scan_output;
      scan_output =
          scan_output == contact_scan_a_.data() ? contact_scan_b_.data() : contact_scan_a_.data();
      if (offset > pair_slot_count / 2) {
        break;
      }
      offset *= 2;
    }
    contact_inclusive_counts_ = scan_input;

    std::uint32_t contact_count = 0;
    check_cuda(cudaMemcpyAsync(&contact_count, contact_inclusive_counts_ + pair_slot_count - 1,
                               sizeof(contact_count), cudaMemcpyDeviceToHost, stream_),
               "failed to download the CUDA contact count");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA contact count or scan failed");
    return contact_count;
  }

  void fill_contacts(std::uint32_t cell_count, const ContactParameters& parameters) {
    const cuda::ContactParametersGpu gpu_parameters{
        .activation_margin = parameters.activation_margin,
        .parallel_sine_threshold = parameters.parallel_sine_threshold,
        .degeneracy_epsilon = parameters.degeneracy_epsilon,
    };
    cuda::launch_contact_fill(
        contact_ids_.data(), contact_centers_.data(), contact_axes_.data(),
        contact_geometry_.data(), contact_counts_.data(), contact_inclusive_counts_,
        contact_first_ids_.data(), contact_second_ids_.data(), contact_first_slots_.data(),
        contact_second_slots_.data(), contact_ordinals_.data(), contact_points_.data(),
        contact_normals_.data(), contact_separations_.data(), contact_weights_.data(),
        gpu_parameters, cell_count, stream_);
    check_cuda(cudaGetLastError(), "failed to launch the CUDA contact-fill kernel");
  }

  [[nodiscard]] ContactGraph download_contacts(std::size_t cell_count,
                                               std::uint32_t contact_count) {
    std::vector<std::uint64_t> first_ids(contact_count);
    std::vector<std::uint64_t> second_ids(contact_count);
    std::vector<std::uint32_t> first_slots(contact_count);
    std::vector<std::uint32_t> second_slots(contact_count);
    std::vector<std::uint32_t> ordinals(contact_count);
    std::vector<float4> points(contact_count);
    std::vector<float4> normals(contact_count);
    std::vector<float> separations(contact_count);
    std::vector<float> weights(contact_count);

    copy_to_host(first_ids, contact_first_ids_, "failed to download CUDA contact first IDs");
    copy_to_host(second_ids, contact_second_ids_, "failed to download CUDA contact second IDs");
    copy_to_host(first_slots, contact_first_slots_, "failed to download CUDA contact first slots");
    copy_to_host(second_slots, contact_second_slots_,
                 "failed to download CUDA contact second slots");
    copy_to_host(ordinals, contact_ordinals_, "failed to download CUDA contact ordinals");
    copy_to_host(points, contact_points_, "failed to download CUDA contact points");
    copy_to_host(normals, contact_normals_, "failed to download CUDA contact normals");
    copy_to_host(separations, contact_separations_, "failed to download CUDA contact separations");
    copy_to_host(weights, contact_weights_, "failed to download CUDA contact weights");
    check_cuda(cudaStreamSynchronize(stream_), "CUDA contact fill or download failed");

    std::vector<CellContact> contacts;
    contacts.reserve(contact_count);
    for (std::uint32_t index = 0; index < contact_count; ++index) {
      if (ordinals[index] > 1) {
        throw std::runtime_error("CUDA contact kernel produced an invalid ordinal");
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

  template <typename T>
  void copy_to_host(std::vector<T>& destination, const CudaBuffer<T>& source,
                    const char* operation) {
    check_cuda(cudaMemcpyAsync(destination.data(), source.data(), destination.size() * sizeof(T),
                               cudaMemcpyDeviceToHost, stream_),
               operation);
  }

  int device_index_{0};
  cudaDeviceProp device_properties_{};
  cudaStream_t stream_{nullptr};

  CudaBuffer<float> lengths_;
  CudaBuffer<float> growth_rates_;

  CudaBuffer<std::uint64_t> contact_ids_;
  CudaBuffer<float4> contact_centers_;
  CudaBuffer<float4> contact_axes_;
  CudaBuffer<float4> contact_geometry_;

  CudaBuffer<std::uint32_t> contact_counts_;
  CudaBuffer<std::uint32_t> contact_scan_a_;
  CudaBuffer<std::uint32_t> contact_scan_b_;
  const std::uint32_t* contact_inclusive_counts_{nullptr};

  CudaBuffer<std::uint64_t> contact_first_ids_;
  CudaBuffer<std::uint64_t> contact_second_ids_;
  CudaBuffer<std::uint32_t> contact_first_slots_;
  CudaBuffer<std::uint32_t> contact_second_slots_;
  CudaBuffer<std::uint32_t> contact_ordinals_;
  CudaBuffer<float4> contact_points_;
  CudaBuffer<float4> contact_normals_;
  CudaBuffer<float> contact_separations_;
  CudaBuffer<float> contact_weights_;
};

}  // namespace

std::unique_ptr<ComputeBackend> make_cuda_backend() { return std::make_unique<CudaBackend>(); }

bool cuda_backend_available() noexcept {
  int device_count = 0;
  const auto result = cudaGetDeviceCount(&device_count);
  if (result != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    return false;
  }
  return device_count > 0;
}

}  // namespace cm2
