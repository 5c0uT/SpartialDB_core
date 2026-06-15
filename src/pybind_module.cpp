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

#ifdef USE_PHYSX
#if USE_PHYSX == 1
    py::class_<PxVec3>(m, "PxVec3")
        .def(py::init<float, float, float>(),
            py::arg("x") = 0.0f, py::arg("y") = 0.0f, py::arg("z") = 0.0f)
        .def_readwrite("x", &PxVec3::x)
        .def_readwrite("y", &PxVec3::y)
        .def_readwrite("z", &PxVec3::z)
        .def("__repr__", [](const PxVec3& vec) {
            return "<PxVec3 x=" + std::to_string(vec.x) +
                   " y=" + std::to_string(vec.y) +
                   " z=" + std::to_string(vec.z) + ">";
        });
#endif
#endif

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
    // ADVANCED QUERY TYPES
    // ========================================================================

    py::class_<NeighborResult>(m, "NeighborResult")
        .def(py::init<>())
        .def_readwrite("objectID", &NeighborResult::objectID)
        .def_readwrite("distance", &NeighborResult::distance)
        .def_readwrite("position", &NeighborResult::position)
        .def("__repr__", [](const NeighborResult& nr) {
            return "<NeighborResult id=" + std::to_string(nr.objectID) +
                   " dist=" + std::to_string(nr.distance) + ">";
        });

    py::class_<RangeQueryResult>(m, "RangeQueryResult")
        .def(py::init<>())
        .def_readwrite("objectID", &RangeQueryResult::objectID)
        .def_readwrite("position", &RangeQueryResult::position)
        .def("__repr__", [](const RangeQueryResult& rqr) {
            return "<RangeQueryResult id=" + std::to_string(rqr.objectID) + ">";
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
        .def("query_ray", [](SpatialDB& self, const PxVec3& origin, const PxVec3& direction, float max_distance) {
            const float origin_values[3] = {origin.x, origin.y, origin.z};
            const float direction_values[3] = {direction.x, direction.y, direction.z};
            return self.queryRay(origin_values, direction_values, max_distance);
        },
            py::arg("origin"), py::arg("direction"), py::arg("max_distance") = 1e6f,
            "Cast single ray and get intersection")
        .def("batch_query_ray", [](SpatialDB& self,
                                   const std::vector<PxVec3>& origins,
                                   const std::vector<PxVec3>& directions,
                                   const std::vector<float>& max_distances) {
            std::vector<float> flat_origins;
            std::vector<float> flat_directions;
            flat_origins.reserve(origins.size() * 3);
            flat_directions.reserve(directions.size() * 3);

            for (const auto& origin : origins) {
                flat_origins.push_back(origin.x);
                flat_origins.push_back(origin.y);
                flat_origins.push_back(origin.z);
            }

            for (const auto& direction : directions) {
                flat_directions.push_back(direction.x);
                flat_directions.push_back(direction.y);
                flat_directions.push_back(direction.z);
            }

            return self.batchQueryRay(flat_origins, flat_directions, max_distances);
        },
            py::arg("origins"), py::arg("directions"), py::arg("max_distances"),
            "Cast multiple rays in batch")
        .def("query_sphere", [](SpatialDB& self, const PxVec3& center, float radius) {
            const float center_values[3] = {center.x, center.y, center.z};
            return self.querySphere(center_values, radius);
        },
            py::arg("center"), py::arg("radius"),
            "Query sphere overlap with geometry")
        .def("query_knn", [](SpatialDB& self, const PxVec3& queryPoint, uint32_t k, float maxRadius) {
            const float point[3] = {queryPoint.x, queryPoint.y, queryPoint.z};
            return self.queryKNN(point, k, maxRadius);
        },
            py::arg("query_point"), py::arg("k"), py::arg("max_radius") = 1e6f,
            "Find k nearest neighbors to a point")
        .def("query_range", [](SpatialDB& self, const PxVec3& minBounds, const PxVec3& maxBounds) {
            const float min_b[3] = {minBounds.x, minBounds.y, minBounds.z};
            const float max_b[3] = {maxBounds.x, maxBounds.y, maxBounds.z};
            return self.queryRange(min_b, max_b);
        },
            py::arg("min_bounds"), py::arg("max_bounds"),
            "Query all points within an axis-aligned bounding box")
        .def("add_point_cloud", [](SpatialDB& self, const std::vector<PxVec3>& points, float voxelSize) {
            std::vector<float> flat;
            flat.reserve(points.size() * 3);
            for (const auto& p : points) {
                flat.push_back(p.x);
                flat.push_back(p.y);
                flat.push_back(p.z);
            }
            self.addPointCloud(flat, voxelSize);
        },
            py::arg("points"), py::arg("voxel_size") = 0.1f,
            "Add a point cloud with optional voxel downsampling");

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

    py::class_<GPUMemoryStats>(m, "GPUMemoryStats")
        .def_readwrite("allocated_bytes", &GPUMemoryStats::allocated_bytes)
        .def_readwrite("peak_bytes", &GPUMemoryStats::peak_bytes)
        .def_readwrite("active_allocations", &GPUMemoryStats::active_allocations)
        .def_readwrite("cuda_device", &GPUMemoryStats::cuda_device)
        .def_readwrite("gpu_available", &GPUMemoryStats::gpu_available)
        .def("__repr__", [](const GPUMemoryStats& s) {
            return "<GPUMemoryStats alloc=" + std::to_string(s.allocated_bytes) +
                   " peak=" + std::to_string(s.peak_bytes) +
                   " device=" + std::to_string(s.cuda_device) + ">";
        });

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
            "Step physics simulation by dt seconds")
        .def("get_memory_stats", &PhysXCore::getMemoryStats,
            "Get GPU memory statistics")
        .def("get_cuda_device", &PhysXCore::getCudaDevice,
            "Get current CUDA device index")
        .def("get_device_count", &PhysXCore::getDeviceCount,
            "Get number of available CUDA devices");

    m.def("init_physx", [](int device) {
        PhysXCore::instance().init(device);
        return true;
    }, py::arg("device") = 0, "Initialize the PhysX runtime");

#endif
#endif

    // ========================================================================
    // MODULE METADATA
    // ========================================================================

    m.attr("__version__") = "0.3.0";
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
