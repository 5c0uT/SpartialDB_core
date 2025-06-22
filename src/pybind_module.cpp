#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SpatialDB.hpp"
#include "CoordinateConverter.hpp"
#include "PhysXCore.hpp"

#include <PxPhysicsAPI.h>

using namespace physx;

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::array<float, 3>);

PYBIND11_MODULE(spatialdb_core, m) {
    m.doc() = "Spatial Database Core Module";

    py::class_<std::array<float, 3>>(m, "FloatArray3")
        .def(py::init<>())
        .def("__getitem__", [](const std::array<float, 3>& arr, size_t i) {
            if (i >= 3) throw py::index_error();
            return arr[i];
        })
        .def("__setitem__", [](std::array<float, 3>& arr, size_t i, float v) {
            if (i >= 3) throw py::index_error();
            arr[i] = v;
        });

    py::class_<RayHit>(m, "RayHit")
        .def(py::init<>())
        .def_readwrite("position", &RayHit::position)
        .def_readwrite("normal", &RayHit::normal)
        .def_readwrite("distance", &RayHit::distance)
        .def_readwrite("objectID", &RayHit::objectID);

    py::class_<SpatialDB>(m, "SpatialDB")
        .def(py::init<>())
        .def("load_las", &SpatialDB::loadLAS)
        .def("build_bvh", &SpatialDB::buildBVH)
        .def("clear_scene", &SpatialDB::clearScene)
        .def("query_ray", &SpatialDB::queryRay)
        .def("batch_query_ray", &SpatialDB::batchQueryRay)
        .def("query_sphere", &SpatialDB::querySphere);

    py::class_<CoordinateConverter>(m, "CoordinateConverter")
        .def(py::init<const std::string&, const std::string&>())
        .def("convert", [](CoordinateConverter& self, double x, double y, double z) {
            self.convert(x, y, z);
            return py::make_tuple(x, y, z);
        });

    py::class_<PhysXCore>(m, "PhysXCore")
        .def_static("instance", &PhysXCore::instance, py::return_value_policy::reference)
        .def("init", &PhysXCore::init)
        .def("physics", &PhysXCore::physics);
}