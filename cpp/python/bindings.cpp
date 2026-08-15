#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "cm2/simulation.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_core, module) {
  module.doc() = "CellModeller2 native simulation core";

  module.def("backend_available", &cm2::backend_available, "backend"_a);

  nb::enum_<cm2::BackendKind>(module, "BackendKind")
      .value("CPU", cm2::BackendKind::cpu)
      .value("METAL", cm2::BackendKind::metal)
      .value("CUDA", cm2::BackendKind::cuda);

  nb::class_<cm2::Vec3>(module, "Vec3")
      .def(nb::init<float, float, float>(), "x"_a = 0.0F, "y"_a = 0.0F, "z"_a = 0.0F)
      .def_rw("x", &cm2::Vec3::x)
      .def_rw("y", &cm2::Vec3::y)
      .def_rw("z", &cm2::Vec3::z);

  nb::class_<cm2::BackendInfo>(module, "BackendInfo")
      .def_ro("kind", &cm2::BackendInfo::kind)
      .def_ro("name", &cm2::BackendInfo::name)
      .def_ro("device", &cm2::BackendInfo::device)
      .def_ro("native", &cm2::BackendInfo::native);

  nb::class_<cm2::CellInit>(module, "CellInit")
      .def(nb::init<>())
      .def_rw("position", &cm2::CellInit::position)
      .def_rw("direction", &cm2::CellInit::direction)
      .def_rw("length", &cm2::CellInit::length)
      .def_rw("radius", &cm2::CellInit::radius)
      .def_rw("growth_rate", &cm2::CellInit::growth_rate)
      .def_rw("cell_type", &cm2::CellInit::cell_type);

  nb::class_<cm2::CellSnapshot>(module, "CellSnapshot")
      .def_ro("id", &cm2::CellSnapshot::id)
      .def_ro("slot", &cm2::CellSnapshot::slot)
      .def_ro("position", &cm2::CellSnapshot::position)
      .def_ro("direction", &cm2::CellSnapshot::direction)
      .def_ro("length", &cm2::CellSnapshot::length)
      .def_ro("radius", &cm2::CellSnapshot::radius)
      .def_ro("growth_rate", &cm2::CellSnapshot::growth_rate)
      .def_ro("cell_type", &cm2::CellSnapshot::cell_type);

  nb::class_<cm2::Simulation>(module, "Simulation")
      .def(nb::init<cm2::BackendKind, std::size_t>(), "backend"_a = cm2::BackendKind::cpu,
           "reserved_capacity"_a = 0)
      .def_prop_ro("backend_info", &cm2::Simulation::backend_info)
      .def_prop_ro("time", &cm2::Simulation::time)
      .def_prop_ro("cell_count", &cm2::Simulation::cell_count)
      .def("add_cell", &cm2::Simulation::add_cell, "cell"_a)
      .def("divide_equal", &cm2::Simulation::divide_equal, "parent_id"_a)
      .def("step", &cm2::Simulation::step, "dt"_a)
      .def("cell", &cm2::Simulation::cell, "id"_a)
      .def("cells", &cm2::Simulation::cells)
      .def("lineage_parent", &cm2::Simulation::lineage_parent, "id"_a)
      .def("validate", &cm2::Simulation::validate);
}
