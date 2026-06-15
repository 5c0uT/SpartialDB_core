# Дизайн: Реализация C++-ядра SpatialDB (этап v0.2.0)

**Дата:** 2026-06-15
**Этап:** v0.2.0 — «Реальное C++-ядро» (выбран из roadmap как следующий этап)
**Статус:** Утверждён пользователем, ожидается переход к плану реализации

---

## 1. Контекст и проблема

SpatialDB_core v0.1.0 декларирует GPU-ускоренные пространственные запросы через PhysX,
но C++-ядро фактически **заглушечное**:

- `SpatialDB::queryRay` всегда возвращает `distance = -1.0` (промах) независимо от геометрии.
- `SpatialDB::querySphere` всегда возвращает пустой список.
- `SpatialDB::loadLAS` и `loadMesh` пустые — в сцену PhysX ничего не добавляется.

Следствие: ни один raycast/sphere-запрос не может найти геометрию, потому что её
просто нет в сцене. Тесты проходят только потому, что проверяют форму результата
(`isinstance(list)`, `hit is not None`), а не содержание (`distance > 0`).

Инфраструктура для решения уже есть, но не используется:
- `BVHManager` создаёт `PxScene`, но `addObject` кладёт `PxBoxGeometry` по AABB
  (неточно — луч проходит сквозь настоящий меш и попадает в бокс).
- `Voxelizer::voxelize` корректно превращает точки в воксельный меш, но никто его не вызывает.

**Цель этапа:** превратить заглушечное ядро в рабочее — после загрузки геометрии
запросы `raycast`/`query_sphere` действительно находят загруженные объекты и
возвращают корректные позиции/расстояния/objectID.

---

## 2. Архитектура

### Принцип

Трилогия ответственности с чёткими границами:

```
SpatialDB (координатор)
   │
   ├── loadLAS(path) ──► GeometryLoader ──► точки ──► Voxelizer ──► MeshData
   ├── loadMesh(path) ──► GeometryLoader ──► MeshData
   ├── addPoints(points) ─────────────────────► Voxelizer ──► MeshData
   ├── addMesh(mesh) ─────────────────────────────────────────► MeshData
   │
   └── всё попадает в ──► BVHManager (владелец PxScene)
                              ├── PxTriangleMesh акторы (с objectID в userdata)
                              └── raycast / overlap — реализация запросов
```

`BVHManager` — единственный владелец `PxScene` и единственный, кто выполняет
физические запросы. `SpatialDB` делегирует ему и загрузку геометрии, и запросы.
`PhysXCore` остаётся синглтоном-инициализатором (без изменений — уже работает).

### Декомпозиция (4 единицы, каждая с одним назначением)

| Компонент | Назначение | Состояние до этапа |
|---|---|---|
| **BVHManager** | Хранить геометрию в PxScene, выполнять raycast/overlap | Существует, но BoxGeometry + нет запросов |
| **GeometryLoader** (новый) | Парсить LAS/OBJ/PLY файлы в массивы вершин/индексов | Отсутствует |
| **Voxelizer** | Точки → воксельный меш | Существует и работает, не подключён |
| **SpatialDB** | Связка: загрузка + делегирование запросов | Заглушки |

---

## 3. Компоненты детально

### 3.1. BVHManager — хранение и запросы

**Изменение:** `PxBoxGeometry` по AABB → `PxTriangleMesh` (точное представление геометрии).

**Новый интерфейс** (`include/BVHManager.hpp`):

```cpp
struct MeshData {
    std::vector<float>    vertices;  // interleaved xyz, длина = 3*N
    std::vector<uint32_t> indices;   // треугольники, длина = 3*M
};

class BVHManager {
public:
    BVHManager();
    ~BVHManager();

    void addMesh(const MeshData& mesh);                       // PxTriangleMesh → RigidStatic
    void addPoints(const std::vector<float>& points, float voxelSize); // Voxelizer → addMesh

    RayHit               raycastSingle(const float origin[3], const float dir[3], float maxDist);
    std::vector<RayHit>  raycastBatch(const float* origins, const float* dirs,
                                      const float* maxDists, size_t n);
    std::vector<uint32_t> querySphere(const float center[3], float radius);

    void buildBVH();   // диагностика (оставить как есть)
    void clearScene(); //释放 всех актёров + сцены
private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};
```

**Реализация запросов через PhysX:**
- `raycastSingle`: `PxScene::raycast(origin, dir, maxDist, PxRaycastBuffer)` →
  при попадании `RayHit{position=hit.position, normal=hit.normal, distance=hit.distance,
  objectID=<индекс актёра из mActors, найденный по hit.actor>}`.
  Промах → `RayHit{distance=-1}` (корректный «нет попадания», не исключение).
- `raycastBatch`: цикл по N лучам, вызывает `raycastSingle`. (PhysX GPU-batch —
  резерв на v0.3; этот этап делает CPU-цикл, что достаточно для корректности.)
- `querySphere`: `PxScene::overlap(PxSphereGeometry(radius), PxTransform(center),
  PxOverlapBuffer)` → собирает `objectID` всех затронутых актёров.

**`objectID`:** индекс актёра в `Impl::mActors`. Для соответствия `hit.actor` →
`objectID` записываем указатель актёра как `userData` шейпа или ищем линейно по
`mActors` (число актёров небольшое — приемлемо). Решение: использовать `userData`
шейпа = индекс в `mActors` (O(1), детерминированно).

**PxTriangleMesh создание:** через отдельный экземпляр `PxCooking`, создаваемый и
владеемый `BVHManager::Impl` (локально, не в PhysXCore). `PxCooking` создаётся
один раз в конструкторе `BVHManager` через `PxCreateCooking(PX_PHYSICS_VERSION,
*foundation, PxCookingParams(scale))` и освобождается в деструкторе. Так
`PhysXCore` не меняется (минимум затронутых компонентов), а BVHManager получает
всё необходимое для `createTriangleMesh(PxTriangleMeshDesc, cooking)`.

### 3.2. GeometryLoader — новый модуль

**Новые файлы:** `include/GeometryLoader.hpp`, `src/GeometryLoader.cpp`.

```cpp
class GeometryLoader {
public:
    static std::vector<float> loadLAS(const std::string& path); // точки xyz
    static MeshData           loadOBJ(const std::string& path); // треугольный меш
    static MeshData           loadPLY(const std::string& path); // ascii + binary_le
};
```

**Собственный LAS-парсер (без внешних зависимостей):** формат LAS 1.2–1.4 имеет
фиксированный binary-хедер (22 байта public header + variable length records +
point data records). Парсер читает header → offset to point data → формат точки
(point data record format 0–10) → извлекает X/Y/Z (масштабируется scale/offset).
~120 строк. Обоснование: загрузка должна идти через нативное ядро (смысл
нативной вокселизации), laspy живёт в Python и не подходит для C++-pipeline.

**OBJ:** парсинг строк `v x y z` и `f i j k` (1-based). Полигоны >3 вершин →
fan-tessellation. Поддержка индексов `v/vt/vn` (берём только вершинные).

**PLY:** заголовок формата определяет layout (ascii / binary_little_endian /
binary_big_endian) и properties. Читаем `vertex` с `float x/y/z` и `face` с
`list uchar int vertex_indices`. Поддержка ascii и binary_little_endian
(распространённые случаи); binary_big_endian — вне этапа (опционально).

### 3.3. Voxelizer

Без изменений — `Voxelizer::voxelize(points, outVerts, outIndices, voxelSize)` уже
работает. Подключается через `BVHManager::addPoints`:
`points → Voxelizer::voxelize → MeshData → addMesh`.

### 3.4. SpatialDB — связка

**Изменения `include/SpatialDB.hpp`:**

```cpp
class SpatialDB {
public:
    SpatialDB();
    ~SpatialDB();

    void loadLAS(const std::string& path, const char* crs = "EPSG:4326");
    void loadMesh(const std::string& path);

    // НОВОЕ — приём геометрии напрямую через API
    void addPoints(const std::vector<float>& points, float voxelSize);
    void addMesh(const MeshData& mesh);

    RayHit               queryRay(const float origin[3], const float dir[3], float maxDist);
    std::vector<RayHit>  batchQueryRay(const std::vector<float>& origins,
                                       const std::vector<float>& directions,
                                       const std::vector<float>& maxDists);
    std::vector<uint32_t> querySphere(const float center[3], float radius);

    void buildBVH();
    void clearScene();

private:
    BVHManager* mBVHManager;
    float       mVoxelSize = 0.1f;   // НОВОЕ — для вокселизации точек
};
```

**Реализация:**

```cpp
void SpatialDB::loadLAS(const std::string& path, const char* crs) {
    auto points = GeometryLoader::loadLAS(path);    // throw при ошибке файла/формата
    mBVHManager->addPoints(points, mVoxelSize);     // Voxelizer → scene
}

void SpatialDB::loadMesh(const std::string& path) {
    // выбор парсера по расширению
    MeshData mesh = endsWith(path, ".obj") ? GeometryLoader::loadOBJ(path)
                  : endsWith(path, ".ply") ? GeometryLoader::loadPLY(path)
                  : throw std::invalid_argument("Unsupported mesh format");
    mBVHManager->addMesh(mesh);
}

RayHit SpatialDB::queryRay(const float o[3], const float d[3], float maxDist) {
    return mBVHManager->raycastSingle(o, d, maxDist);
}
// batchQueryRay, querySphere — аналогично делегируют.
```

`crs` в `loadLAS` принимается, но **не применяется** — преобразование координат
отдельный пункт roadmap (v0.3). Фиксируется как известное ограничение.

### 3.5. pybind + Python обвязка

В `src/pybind_module.cpp`:
- Забиндить `SpatialDB::addPoints` (`add_points`), `SpatialDB::addMesh` (`add_mesh`).
- `add_points` принимает Python list/`numpy.ndarray` (N,3) → flatten в `vector<float>`.
- `add_mesh` принимает объект `{vertices: list/ndarray, indices: list/ndarray}`.

В `python/spatial_db/core.py`:
- `SpatialDB.add_points(self, points: ndarray, voxel_size: float)` — обёртка над
  нативным методом, с валидацией формы `(N,3)`.
- `SpatialDB.add_mesh(self, vertices: ndarray, indices: ndarray)` — валидация
  `(N,3)` и `(M,3)`.
- Существующие `load_las`/`load_mesh` (Python) при наличии нативного модуля
  делегируют в `self.core.load_las(path)` / `self.core.load_mesh(path)` — нативный
  pipeline. Stub-fallback остаётся для случая без нативного модуля.

---

## 4. Поток данных

```
Путь A — точки (LAS / add_points):
  источник ──► GeometryLoader ──► vector<float> xyz
                                   └► Voxelizer.voxelize(points, voxelSize)
                                          └► MeshData {verts, indices}
                                                └► BVHManager::addMesh ──► PxTriangleMesh ──► PxScene

Путь B — меши (OBJ/PLY / add_mesh):
  источник ──► GeometryLoader ──► MeshData {verts, indices}
                                          └► BVHManager::addMesh ──► PxTriangleMesh ──► PxScene

Запросы:
  queryRay/batch ──► BVHManager ──► PxScene::raycast ──► RayHit{pos,normal,dist,objectID}
  querySphere    ──► BVHManager ──► PxScene::overlap(PxSphereGeometry) ──► vector<objectID>
```

`Voxelizer` задействуется только для точек; меши идут напрямую. Так `.las`-облако
становится воксельным мешом, а `queryRay` находит конкретный воксель.

---

## 5. Обработка ошибок (минимум для рабочей логики)

Полный error-handling — отдельный пункт roadmap; здесь ровно столько, сколько
нужно для корректного поведения:

| Ситуация | Поведение |
|---|---|
| Файл не существует / пустой | `throw std::runtime_error` из Loader → в Python `RuntimeError` |
| Неверный формат (magic / header) | `throw std::runtime_error("Invalid LAS/OBJ/PLY")` |
| `addMesh`/`addPoints` с пустыми данными | `throw std::invalid_argument` |
| Неподдерживаемое расширение меша | `throw std::invalid_argument("Unsupported mesh format")` |
| PhysX init / createMesh / cooking failed | `throw std::runtime_error` |
| Raycast, но в сцене нет геометрии | вернуть `RayHit{distance=-1}` (промах, **не** ошибка) |

C++-исключения пробрасываются в Python автоматически через pybind11.
В Python-обёртке пути **загрузки** исключения **не глотаются** (отличие от текущего
`raycast`, где catch оставляется — там промах ≠ ошибка).

---

## 6. Тестирование (TDD)

Новые тесты добавляются к существующим 52. Существующие не ломаются
(stub-fallback сохраняется для случая без нативного модуля).

### 6.1. BVHManager (через Python)

- `addMesh` (плоскость z=0, 2 треугольника) → `raycastSingle` лучом сверху вниз →
  `hit.distance ≈ z_источника`, `hit.position.z ≈ 0`, `hit.objectID == 0`.
- Луч мимо геометрии → `distance == -1`.
- `querySphere` по центру куба → возвращает `[0]`; далеко → `[]`.

### 6.2. GeometryLoader

- `loadOBJ` синтетический квадрат → 4 вершины, 6 индексов.
- `loadLAS` мини-LAS (100 точек, генерируется в тесте) → 300 floats.
- `loadPLY` ascii-вариант и binary_little_endian-вариант.
- Несуществующий файл → `pytest.raises(RuntimeError)`.

### 6.3. Интеграционные (сквозь Python) — ключевые

- `db.add_mesh(...)` → `db.raycast(...)` → **реальный hit** (`distance > 0`).
  Главный тест этапа: доказывает, что ядро больше не заглушка.
- `db.load_las(test.las)` → вокселизация → `db.query_sphere(...)` → находит objectID.
- `db.batch_raycast(...)` на загруженной геометрии → корректные hits.

### 6.4. Тестовые данные

Мини `.las`/`.obj`/`.ply` генерируются в `tests/conftest.py` (pytest fixtures)
или складываются в `tests/data/`. Никаких внешних загрузок.

---

## 7. Верификация

```bash
# Полная сборка + smoke импорт + pytest
powershell -ExecutionPolicy Bypass -File .\run.ps1 -SkipPip -SkipVcpkg

# Только тесты
conda run -n spatial_env python -m pytest tests -v --tb=short
```

**Критерий успеха этапа:**
1. Все 52 существующих теста PASS.
2. ~15 новых тестов PASS.
3. Ключевой интеграционный тест «raycast на загруженной геометрии даёт hit
   с `distance > 0`» — зелёный.

---

## 8. Что НЕ входит в этап (явные границы)

- **CRS-преобразование** при загрузке — `loadLAS` принимает `crs`, но не применяет
  (v0.3 roadmap).
- **Полная обработка ошибок** (иерархия `SpatialDBError`, убрать глотание ошибок
  в Python) — отдельный пункт v0.2.0 roadmap.
- **Linux-поддержка** — отдельный пункт v0.2.0 roadmap.
- **GPU-batch raycast** (`PxCudaContextManager`, mass-ray) — v0.3 (оптимизация).
- **Binary big-endian PLY** — опционально, вне этапа.
- **Синхронизация версий** (`setup.py`=0.1.0 vs pybind `3.0.0`) — вне этого этапа.

---

## 9. Риски

| Риск | Смягчение |
|---|---|
| `PxCooking` требует корректного `PxCookingParams` с тем же `TolerancesScale`, что и `PxPhysics` | Передавать `physics->getTolerancesScale()` в `PxCookingParams` при создании (зафиксировано в §3.1) |
| LAS point data record format различается (0–10) | Поддержать форматы 0–3 (охватывает >95% файлов), остальные — `throw` с понятным сообщением |
| OBJ с полигонами >3 вершин | Fan-tessellation (указано в дизайне) |
| Существующие тесты ожидают stub-поведение | Проверить: тесты проверяют форму, не содержание — должны остаться зелёными |

---

## 10. Структура файлов (изменения)

```
include/
  GeometryLoader.hpp   (НОВОЕ)
  BVHManager.hpp       (изменить: MeshData, addMesh/addPoints, запросы)
  SpatialDB.hpp        (изменить: addPoints/addMesh, mVoxelSize)
  Voxelizer.hpp        (без изменений)
  PhysXCore.hpp        (возможно: добавить getCooking())
src/
  GeometryLoader.cpp   (НОВОЕ)
  BVHManager.cpp       (переработать: TriangleMesh + raycast/overlap)
  SpatialDB.cpp        (связка, делегирование)
  pybind_module.cpp    (биндинг add_points/add_mesh)
  Voxelizer.cpp        (без изменений)
  PhysXCore.cpp        (возможно: PxCooking)
CMakeLists.txt         (добавить GeometryLoader.cpp в CORE_SOURCES)
python/spatial_db/core.py  (add_points/add_mesh обёртки, load_* делегирование)
tests/
  test_spatialdb.py    (новые тесты)
  conftest.py          (fixtures: синтетические .las/.obj/.ply)
  data/                (опционально: готовые мини-файлы)
```
