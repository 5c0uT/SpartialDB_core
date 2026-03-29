import sys


def pytest_collection_modifyitems(session, config, items):
    import spatial_db as spatial_db_module

    for module_name in ("test_spatialdb", "tests.test_spatialdb"):
        test_module = sys.modules.get(module_name)
        if test_module is None:
            continue

        fixture_obj = getattr(test_module, "spatial_db", None)
        if fixture_obj is None:
            continue

        fixture_obj.HAS_NATIVE_MODULE = spatial_db_module.HAS_NATIVE_MODULE
        fixture_obj.native = spatial_db_module.native
