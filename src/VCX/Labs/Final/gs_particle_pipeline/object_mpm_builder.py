from dataclasses import dataclass

import numpy as np
import taichi as ti

from .particles import GaussianHostData, GaussianParticleSet


@dataclass
class ObjectMPMBuilderConfig:
    max_particles: int | None = None
    fill_internal: bool = True
    fill_grid: int = 32
    surface_ratio: float = 0.65
    density: float = 200.0
    youngs_modulus: float = 1.0e4
    poisson_ratio: float = 0.40
    pin_mode: str = "bottom"
    pin_thickness: float = 0.08
    density_threshold: float = 0.02
    search_threshold: float = 0.01
    max_particles_per_cell: int = 1


@dataclass
class ElasticObjectRenderConfig:
    max_render_particles: int


def _uniform_sample_indices(count: int, budget: int | None) -> np.ndarray:
    if budget is None or count <= budget:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(0)
    return rng.choice(count, size=budget, replace=False).astype(np.int64)


def _dilate_occupancy(occupied: np.ndarray) -> np.ndarray:
    padded = np.pad(occupied, 1, mode="constant", constant_values=False)
    dilated = occupied.copy()
    for dx in range(3):
        for dy in range(3):
            for dz in range(3):
                dilated |= padded[dx : dx + occupied.shape[0], dy : dy + occupied.shape[1], dz : dz + occupied.shape[2]]
    return dilated


def _flood_external(empty: np.ndarray) -> np.ndarray:
    shape = empty.shape
    external = np.zeros(shape, dtype=bool)
    stack: list[tuple[int, int, int]] = []

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in (0, shape[2] - 1):
                if empty[i, j, k] and not external[i, j, k]:
                    external[i, j, k] = True
                    stack.append((i, j, k))
    for i in range(shape[0]):
        for k in range(shape[2]):
            for j in (0, shape[1] - 1):
                if empty[i, j, k] and not external[i, j, k]:
                    external[i, j, k] = True
                    stack.append((i, j, k))
    for j in range(shape[1]):
        for k in range(shape[2]):
            for i in (0, shape[0] - 1):
                if empty[i, j, k] and not external[i, j, k]:
                    external[i, j, k] = True
                    stack.append((i, j, k))

    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    while stack:
        i, j, k = stack.pop()
        for di, dj, dk in neighbors:
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:
                if empty[ni, nj, nk] and not external[ni, nj, nk]:
                    external[ni, nj, nk] = True
                    stack.append((ni, nj, nk))
    return external


def _nearest_indices(points: np.ndarray, surface: np.ndarray, chunk_size: int = 1024) -> np.ndarray:
    nearest = np.zeros((points.shape[0],), dtype=np.int64)
    for start in range(0, points.shape[0], chunk_size):
        end = min(start + chunk_size, points.shape[0])
        diff = points[start:end, None, :] - surface[None, :, :]
        nearest[start:end] = np.argmin(np.sum(diff * diff, axis=2), axis=1)
    return nearest


def _knn_indices_and_weights(points: np.ndarray, surface: np.ndarray, k: int = 4, chunk_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    k = min(max(1, int(k)), surface.shape[0])
    all_indices = np.zeros((points.shape[0], k), dtype=np.int32)
    all_weights = np.zeros((points.shape[0], k), dtype=np.float32)
    for start in range(0, points.shape[0], chunk_size):
        end = min(start + chunk_size, points.shape[0])
        diff = points[start:end, None, :] - surface[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
        d = np.take_along_axis(dist2, idx, axis=1)
        w = 1.0 / np.maximum(d, 1e-10)
        w = w / np.sum(w, axis=1, keepdims=True)
        all_indices[start:end] = idx.astype(np.int32)
        all_weights[start:end] = w.astype(np.float32)
    return all_indices, all_weights


def _covariance_to_inverse(covariance: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(covariance.astype(np.float64))
    eigvals = np.maximum(eigvals, 1e-10)
    inv_cov = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
    return inv_cov.astype(np.float32)


def _build_density_grid(
    position: np.ndarray,
    covariance: np.ndarray,
    opacity: np.ndarray,
    fill_grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    grid = max(int(fill_grid), 4)
    bbox_min = position.min(axis=0)
    bbox_max = position.max(axis=0)
    extent = np.maximum(bbox_max - bbox_min, 1e-4)
    pad = np.maximum(extent * 0.03, 1e-4)
    bbox_min = bbox_min - pad
    bbox_max = bbox_max + pad
    extent = bbox_max - bbox_min
    cell = extent / float(grid)
    cell_size = float(np.max(cell))

    normalized = (position - bbox_min[None, :]) / extent[None, :]
    voxels = np.clip((normalized * grid).astype(np.int32), 0, grid - 1)
    grid_count = np.zeros((grid, grid, grid), dtype=np.int32)
    np.add.at(grid_count, (voxels[:, 0], voxels[:, 1], voxels[:, 2]), 1)

    grid_density = np.zeros((grid, grid, grid), dtype=np.float32)
    for p, cov, alpha in zip(position, covariance, opacity):
        inv_cov = _covariance_to_inverse(cov)
        eigvals = np.maximum(np.linalg.eigvalsh(cov), 1e-10)
        radius = max(1, int(np.ceil(3.0 * float(np.sqrt(np.max(eigvals))) / max(cell_size, 1e-8))))
        base = np.clip(((p - bbox_min) / extent * grid).astype(np.int32), 0, grid - 1)
        lo = np.maximum(base - radius, 0)
        hi = np.minimum(base + radius + 1, grid)
        for i in range(lo[0], hi[0]):
            for j in range(lo[1], hi[1]):
                for k in range(lo[2], hi[2]):
                    center = bbox_min + (np.array([i, j, k], dtype=np.float32) + 0.5) * cell
                    diff = p - center
                    grid_density[i, j, k] += float(alpha) * float(np.exp(-0.5 * diff @ inv_cov @ diff))
    return grid_density, grid_count, bbox_min.astype(np.float32), cell.astype(np.float32), float(np.prod(cell))


def _ray_hits_surface(dense: np.ndarray, index: np.ndarray, axis: int, sign: int) -> bool:
    pos = index.copy()
    pos[axis] += sign
    while np.all(pos >= 0) and np.all(pos < np.array(dense.shape)):
        if dense[tuple(pos)]:
            return True
        pos[axis] += sign
    return False


def _ray_crossing_count(dense: np.ndarray, index: np.ndarray, axis: int, sign: int) -> int:
    pos = index.copy()
    pos[axis] += sign
    crossings = 0
    prev = bool(dense[tuple(index)])
    while np.all(pos >= 0) and np.all(pos < np.array(dense.shape)):
        cur = bool(dense[tuple(pos)])
        if cur != prev and not prev:
            crossings += 1
        prev = cur
        pos[axis] += sign
    return crossings


def _make_density_filled_points(
    surface_position: np.ndarray,
    surface_covariance: np.ndarray,
    opacity: np.ndarray,
    fill_grid: int,
    count: int | None,
    density_threshold: float,
    search_threshold: float,
    max_particles_per_cell: int,
) -> tuple[np.ndarray, float]:
    if count == 0 or surface_position.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), 1e-6

    grid_density, grid_count, bbox_min, cell, cell_volume = _build_density_grid(
        surface_position, surface_covariance, opacity, fill_grid
    )
    dense = grid_density > float(density_threshold)
    search_dense = grid_density > float(search_threshold)

    rng = np.random.default_rng(1)
    points: list[np.ndarray] = []
    ppc = max(1, int(max_particles_per_cell))

    for i, j, k in np.argwhere(dense):
        need = max(0, ppc - int(grid_count[i, j, k]))
        for _ in range(need):
            points.append(bbox_min + (np.array([i, j, k], dtype=np.float32) + rng.random(3).astype(np.float32)) * cell)

    for index in np.argwhere(~dense):
        if grid_count[tuple(index)] > 0:
            continue
        hit_all = True
        for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
            if not _ray_hits_surface(search_dense, index, axis, sign):
                hit_all = False
                break
        if hit_all and _ray_crossing_count(search_dense, index, 2, 1) % 2 == 1:
            for _ in range(ppc):
                points.append(bbox_min + (index.astype(np.float32) + rng.random(3).astype(np.float32)) * cell)

    if not points:
        margin = max(1, int(fill_grid) // 6)
        indices = []
        for i in range(margin, int(fill_grid) - margin):
            for j in range(margin, int(fill_grid) - margin):
                for k in range(margin, int(fill_grid) - margin):
                    if grid_count[i, j, k] == 0:
                        indices.append((i, j, k))
        for i, j, k in indices:
            points.append(bbox_min + (np.array([i, j, k], dtype=np.float32) + rng.random(3).astype(np.float32)) * cell)

    points_array = np.asarray(points, dtype=np.float32)

    if count is not None and points_array.shape[0] > count:
        sample = _uniform_sample_indices(points_array.shape[0], count)
        points_array = points_array[sample]

    return points_array.astype(np.float32), cell_volume


def _compute_particle_volumes(position: np.ndarray, fill_grid: int) -> np.ndarray:
    if position.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    grid = max(int(fill_grid), 4)
    bbox_min = position.min(axis=0)
    bbox_max = position.max(axis=0)
    extent = np.maximum(bbox_max - bbox_min, 1e-4)
    pad = np.maximum(extent * 0.03, 1e-4)
    bbox_min = bbox_min - pad
    bbox_max = bbox_max + pad
    extent = bbox_max - bbox_min
    cell = extent / float(grid)
    cell_volume = float(np.prod(cell))
    voxels = np.clip(((position - bbox_min[None, :]) / extent[None, :] * grid).astype(np.int32), 0, grid - 1)
    counts = np.zeros((grid, grid, grid), dtype=np.int32)
    np.add.at(counts, (voxels[:, 0], voxels[:, 1], voxels[:, 2]), 1)
    volume = np.empty((position.shape[0],), dtype=np.float32)
    for idx, voxel in enumerate(voxels):
        volume[idx] = cell_volume / max(int(counts[tuple(voxel)]), 1)
    return volume


def _pin_by_region(position: np.ndarray, mode: str, thickness: float) -> np.ndarray:
    pinned = np.zeros((position.shape[0],), dtype=np.int32)
    if mode in {"none", "ply"} or position.shape[0] == 0:
        return pinned

    bbox_min = position.min(axis=0)
    bbox_max = position.max(axis=0)
    extent = np.maximum(bbox_max - bbox_min, 1e-6)
    band = np.maximum(extent * max(float(thickness), 0.0), 1e-6)

    if mode == "bottom":
        pinned[position[:, 1] <= bbox_min[1] + band[1]] = 1
    elif mode == "top":
        pinned[position[:, 1] >= bbox_max[1] - band[1]] = 1
    elif mode == "left":
        pinned[position[:, 0] <= bbox_min[0] + band[0]] = 1
    elif mode == "right":
        pinned[position[:, 0] >= bbox_max[0] - band[0]] = 1
    else:
        raise ValueError(f"Unsupported elastic pin mode: {mode}")
    return pinned


def build_elastic_object_host_data(source: GaussianHostData, config: ObjectMPMBuilderConfig) -> GaussianHostData:
    """Build an elastic MPM-ready Gaussian set from visual 3DGS data.

    The returned set uses one shared particle representation for simulation and
    rendering. Surface Gaussians are budgeted first, and remaining capacity is
    filled with interior particles so elastic MPM has volume support.
    """

    max_particles = None if config.max_particles is None else max(1, int(config.max_particles))
    fill_enabled = bool(config.fill_internal)
    if max_particles is None:
        surface_budget = source.position.shape[0]
    elif fill_enabled:
        surface_budget = max(1, min(source.position.shape[0], int(max_particles * config.surface_ratio)))
    else:
        surface_budget = min(source.position.shape[0], max_particles)

    surface_indices = _uniform_sample_indices(source.position.shape[0], surface_budget)
    position = source.position[surface_indices].astype(np.float32)
    base_covariance = source.base_covariance[surface_indices].astype(np.float32)
    opacity = source.opacity[surface_indices].astype(np.float32)
    sh_coefficients = source.sh_coefficients[surface_indices].astype(np.float32)
    scale = source.scale[surface_indices].astype(np.float32)
    rotation = source.rotation[surface_indices].astype(np.float32)

    fill_capacity = None if max_particles is None else max_particles - position.shape[0]
    internal_position, cell_volume = _make_density_filled_points(
        position,
        base_covariance,
        opacity,
        config.fill_grid,
        fill_capacity,
        config.density_threshold,
        config.search_threshold,
        config.max_particles_per_cell,
    ) if fill_enabled else (
        np.zeros((0, 3), dtype=np.float32),
        1e-6,
    )

    internal_nearest = np.zeros((0,), dtype=np.int64)
    if internal_position.shape[0] > 0:
        internal_nearest = _nearest_indices(internal_position, position)
        radius = max(float((3.0 * cell_volume / (4.0 * np.pi)) ** (1.0 / 3.0)), 1e-5)
        internal_covariance = np.tile(np.eye(3, dtype=np.float32) * (radius * radius), (internal_position.shape[0], 1, 1))
        internal_scale = np.full((internal_position.shape[0], 3), radius, dtype=np.float32)
        internal_rotation = np.zeros((internal_position.shape[0], 4), dtype=np.float32)
        internal_rotation[:, 0] = 1.0

        position = np.concatenate([position, internal_position], axis=0)
        base_covariance = np.concatenate([base_covariance, internal_covariance], axis=0)
        opacity = np.concatenate([opacity, opacity[internal_nearest]], axis=0)
        sh_coefficients = np.concatenate([sh_coefficients, sh_coefficients[internal_nearest]], axis=0)
        scale = np.concatenate([scale, internal_scale], axis=0)
        rotation = np.concatenate([rotation, internal_rotation], axis=0)

    count = position.shape[0]
    volume = np.maximum(_compute_particle_volumes(position, config.fill_grid), 1e-9).astype(np.float32)
    density = np.full((count,), float(config.density), dtype=np.float32)
    mass = density * volume
    youngs_modulus = np.full((count,), float(config.youngs_modulus), dtype=np.float32)
    poisson_ratio = np.full((count,), float(config.poisson_ratio), dtype=np.float32)
    material_id = np.zeros((count,), dtype=np.int32)
    pinned = _pin_by_region(position, config.pin_mode, config.pin_thickness)

    if source.material_id is not None:
        surface_material = source.material_id[surface_indices].astype(np.int32)
        material_id[:surface_budget] = surface_material
        if internal_nearest.shape[0] > 0:
            material_id[surface_budget:] = surface_material[internal_nearest]
    if config.pin_mode == "ply" and source.pinned is not None:
        pinned = np.zeros((count,), dtype=np.int32)
        surface_pinned = source.pinned[surface_indices].astype(np.int32)
        pinned[:surface_budget] = surface_pinned
        if internal_nearest.shape[0] > 0:
            pinned[surface_budget:] = surface_pinned[internal_nearest]

    return GaussianHostData(
        position=position,
        base_covariance=base_covariance,
        opacity=opacity,
        sh_coefficients=sh_coefficients,
        scale=scale,
        rotation=rotation,
        mass=mass,
        volume=volume,
        material_id=material_id,
        density=density,
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
        pinned=pinned,
    )


def build_elastic_render_host_data(source: GaussianHostData, config: ElasticObjectRenderConfig) -> GaussianHostData:
    render_count = min(source.position.shape[0], max(1, int(config.max_render_particles)))
    indices = _uniform_sample_indices(source.position.shape[0], render_count)
    mass = np.ones((render_count,), dtype=np.float32)
    volume = np.maximum(np.linalg.det(source.base_covariance[indices]), 1e-12).astype(np.float32)
    return GaussianHostData(
        position=source.position[indices].astype(np.float32),
        base_covariance=source.base_covariance[indices].astype(np.float32),
        opacity=source.opacity[indices].astype(np.float32),
        sh_coefficients=source.sh_coefficients[indices].astype(np.float32),
        scale=source.scale[indices].astype(np.float32),
        rotation=source.rotation[indices].astype(np.float32),
        mass=mass,
        volume=volume,
    )


@ti.data_oriented
class ElasticObjectRenderLayer:
    """Bind high-density visual Gaussians to elastic MPM particles.

    Unlike cloth dense rendering, the source Gaussians are unstructured. Each
    visual Gaussian is attached to its nearest simulation particle in the rest
    state, then follows that particle's local deformation gradient.
    """

    def __init__(self, sim_particles: GaussianParticleSet, render_particles: GaussianParticleSet):
        self.sim_particles = sim_particles
        self.render_particles = render_particles
        render_pos = render_particles.rest_position.to_numpy()[: render_particles.count[None]]
        sim_pos = sim_particles.rest_position.to_numpy()[: sim_particles.count[None]]
        knn, weights = _knn_indices_and_weights(render_pos, sim_pos, k=4)
        rest_offsets = render_pos[:, None, :] - sim_pos[knn]

        self.k = 4
        self.nearest = ti.field(dtype=ti.i32, shape=(render_particles.max_particles, self.k))
        self.weight = ti.field(dtype=ti.f32, shape=(render_particles.max_particles, self.k))
        self.rest_offset = ti.Vector.field(3, dtype=ti.f32, shape=(render_particles.max_particles, self.k))

        nearest_full = np.zeros((render_particles.max_particles, self.k), dtype=np.int32)
        weight_full = np.zeros((render_particles.max_particles, self.k), dtype=np.float32)
        offset_full = np.zeros((render_particles.max_particles, self.k, 3), dtype=np.float32)
        nearest_full[: knn.shape[0], : knn.shape[1]] = knn
        weight_full[: weights.shape[0], : weights.shape[1]] = weights
        offset_full[: rest_offsets.shape[0], : rest_offsets.shape[1]] = rest_offsets.astype(np.float32)
        self.nearest.from_numpy(nearest_full)
        self.weight.from_numpy(weight_full)
        self.rest_offset.from_numpy(offset_full)

    @ti.kernel
    def update(self):
        for p in range(self.render_particles.count[None]):
            pos = ti.Vector([0.0, 0.0, 0.0])
            cov = ti.Matrix.zero(ti.f32, 3, 3)
            for n in ti.static(range(4)):
                sim_id = self.nearest[p, n]
                w = self.weight[p, n]
                f = self.sim_particles.deformation_gradient[sim_id]
                pos += w * (self.sim_particles.position[sim_id] + f @ self.rest_offset[p, n])
                cov += w * (f @ self.render_particles.base_covariance[p] @ f.transpose())
            self.render_particles.position[p] = pos
            self.render_particles.current_covariance[p] = cov
