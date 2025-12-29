#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SpatialDB.hpp"

// Условные включения в зависимости от конфигурации сборки
#ifdef USE_PROJ
#if USE_PROJ == 1
#include "CoordinateConverter.hpp"
#endif
#endif

#ifdef USE_PHYSX
#if USE_PHYSX == 1
#include "PhysXCore.hpp"
#include <PxPhysicsAPI.h>
using namespace physx;
#endif
#endif

namespace py = pybind11;

// ============================================================================
// OPAQUE TYPES
// ============================================================================

PYBIND11_MAKE_OPAQUE(std::array<float, 3>);

// ============================================================================
// PYBIND11 MODULE
// ============================================================================

PYBIND11_MODULE(spatialdb_core_pybind, m) {
    m.doc() = "Spatial Database Core Module - Advanced Spatial Indexing with Physics";

    // ========================================================================
    // BASIC TYPES
    // ========================================================================

    py::class_<std::array<float, 3>>(m, "FloatArray3")        
        .def(py::init<>())
        .def("__getitem__", [](const std::array<float, 3>& arr, size_t i) {
            if (i >= 3) throw py::index_error();
            return arr[i];
        })
        .def("__setitem__", [](std::array<float, 3>& arr, size_t i, float v) {
            if (i >= 3) throw py::index_error();
            arr[i] = v;
        })
        .def("__repr__", [](const std::array<float, 3>& arr) {
            return "[" + std::to_string(arr[0]) + ", " + 
                   std::to_string(arr[1]) + ", " + 
                   std::to_string(arr[2]) + "]";
        });

    py::class_<RayHit>(m, "RayHit")
        .def(py::init<>())
        .def_readwrite("position", &RayHit::position)
        .def_readwrite("normal", &RayHit::normal)
        .def_readwrite("distance", &RayHit::distance)
        .def_readwrite("objectID", &RayHit::objectID)
        .def("__repr__", [](const RayHit& hit) {
            return std::string("<RayHit distance=") + std::to_string(hit.distance) + 
                   " objectID=" + std::to_string(hit.objectID) + ">";
        });

    // ========================================================================
    // SPATIALDB CLASS
    // ========================================================================

    py::class_<SpatialDB>(m, "SpatialDB")
        .def(py::init<>())
        .def("load_las", &SpatialDB::loadLAS,
            py::arg("path"), py::arg("crs") = "EPSG:4326",
            "Load LAS/LAZ point cloud file with optional CRS")
        .def("build_bvh", &SpatialDB::buildBVH,
            "Build BVH acceleration structure")
        .def("clear_scene", &SpatialDB::clearScene,
            "Clear all loaded geometry")
        .def("query_ray", &SpatialDB::queryRay,
            py::arg("origin"), py::arg("direction"), py::arg("max_distance") = 1e6f,
            "Cast single ray and get intersection")
        .def("batch_query_ray", &SpatialDB::batchQueryRay,
            py::arg("origins"), py::arg("directions"), py::arg("max_distances"),
            "Cast multiple rays in batch")
        .def("query_sphere", &SpatialDB::querySphere,
            py::arg("center"), py::arg("radius"),
            "Query sphere overlap with geometry");

    // ========================================================================
    // COORDINATE CONVERTER (Conditional)
    // ========================================================================

#ifdef USE_PROJ
#if USE_PROJ == 1

    py::class_<CoordinateConverter>(m, "CoordinateConverter") 
        .def(py::init<const std::string&, const std::string&>(),
            py::arg("from_crs"), py::arg("to_crs"),
            "Create coordinate converter between two CRS")
        .def("convert", [](CoordinateConverter& self, double x, double y, double z) {
            self.convert(x, y, z);
            return py::make_tuple(x, y, z);
        },
        py::arg("x"), py::arg("y"), py::arg("z"),
        "Convert single coordinate point")
        .def("convert_batch", [](CoordinateConverter& self, 
                                  const std::vector<std::array<double, 3>>& coords) {
            std::vector<std::array<double, 3>> result = coords;
            for(auto& coord : result) {
                self.convert(coord[0], coord[1], coord[2]);
            }
            return result;
        },
        py::arg("coordinates"),
        "Convert batch of coordinate points");

#endif
#endif

    // ========================================================================
    // PHYSX CORE (Conditional)
    // ========================================================================

#ifdef USE_PHYSX
#if USE_PHYSX == 1

    py::class_<PhysXCore>(m, "PhysXCore")
        .def_static("instance", &PhysXCore::instance, 
            py::return_value_policy::reference,
            "Get singleton instance of PhysXCore")
        .def("init", &PhysXCore::init,
            "Initialize PhysX engine")
        .def("cleanup", &PhysXCore::cleanup,
            "Cleanup PhysX engine")
        .def("physics", &PhysXCore::physics,
            py::return_value_policy::reference,
            "Get PxPhysics pointer")
        .def("create_scene", &PhysXCore::createScene,
            "Create a new physics scene")
        .def("step_simulation", &PhysXCore::stepSimulation,
            py::arg("dt"),
            "Step physics simulation by dt seconds");

#endif
#endif

    // ========================================================================
    // MODULE METADATA
    // ========================================================================

    m.attr("__version__") = "3.0.0";
    m.attr("__author__") = "SpatialDB Team";
    m.attr("__license__") = "MIT";
    
    // Build configuration flags
    m.attr("HAS_PHYSX") = 
#ifdef USE_PHYSX
#if USE_PHYSX == 1
        true
#else
        false
#endif
#else
        false
#endif
    ;
    
    m.attr("HAS_PROJ") = 
#ifdef USE_PROJ
#if USE_PROJ == 1
        true
#else
        false
#endif
#else
        false
#endif
    ;
    
    m.attr("HAS_CURL") = 
#ifdef USE_CURL
#if USE_CURL == 1
        true
#else
        false
#endif
#else
        false
#endif
    ;
}
