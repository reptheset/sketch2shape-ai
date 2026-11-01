"""
Stage 7: mesh validation and light, material-safe clean-up with trimesh.

Input:  model_raw.glb, exactly as returned by Stage 6 (TRELLIS or TripoSR).
Output: model.glb        the cleaned, normalised asset served to the viewer and download button
        validation.json  the full report (also stored in details.validation)

Checks (the job fails only on the hard ones):
  hard  - the file loads as a glTF scene
        - it contains at least one triangle mesh with vertices and faces
        - all vertex coordinates are finite (no NaN / infinity)
        - the object has a real size (not collapsed to a point, line or flat sheet)
  soft  - watertight, consistent winding, has texture/colour, triangle count
          (reported as warnings; single-view models are often open at the bottom or
          back, and that is fine for previews and most engines, so watertightness is NOT required)

Clean-up (never touches UVs, textures or materials):
  - removes non-mesh items (stray points/lines), duplicate and zero-area triangles,
    and vertices no triangle uses
  - optionally normalises the model: centred on X/Z, resting on the ground (Y = 0),
    largest side = MESH_TARGET_SIZE metres (glTF units), so every model opens the same way

Safety net: the cleaned file is loaded again and compared with the original (meshes,
textures, vertex colours, triangle count). If anything was lost in re-export, the original
file is shipped unchanged instead, so materials are always preserved.
"""

import asyncio
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.errors import PipelineError

logger = logging.getLogger("sketch2shape.validation")

MIN_FACES = 20
MIN_VERTICES = 10
MIN_THICKNESS_RATIO = 1e-3  # smallest side vs. largest side; below this the "object" is a flat sheet
LARGE_FACE_COUNT = 500_000


@dataclass
class MeshStats:
    meshes: int = 0
    vertices: int = 0
    faces: int = 0
    textured_meshes: int = 0
    vertex_colour_meshes: int = 0
    materials: list[str] = field(default_factory=list)
    texture_sizes: list[list[int]] = field(default_factory=list)


@dataclass
class ValidationResult:
    glb_path: Path
    valid: bool
    used_original: bool  # True when the cleaned re-export was rejected and the raw file was kept
    before: MeshStats
    after: MeshStats
    watertight: bool
    winding_consistent: bool
    extents: list[float]
    original_extents: list[float]
    scale_factor: float
    removed: dict[str, int]
    surface_area: float
    volume: float | None
    file_size_bytes: int
    warnings: list[str]
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["glb_path"] = self.glb_path.name
        return data


# ---------------------------------------------------------------- helpers


def _load_scene(path: Path) -> Any:
    import trimesh

    try:
        # process=False: no automatic vertex merging, so UV seams and colours stay exactly as made.
        return trimesh.load(str(path), force="scene", process=False)
    except Exception as exc:
        raise PipelineError(f"The 3D file could not be read: {str(exc)[:200]}") from exc


def _texture_image(material: Any) -> Any:
    return getattr(material, "baseColorTexture", None) or getattr(material, "image", None)


def _stats(scene: Any) -> MeshStats:
    import trimesh

    stats = MeshStats()
    for mesh in scene.geometry.values():
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        stats.meshes += 1
        stats.vertices += len(mesh.vertices)
        stats.faces += len(mesh.faces)
        visual = mesh.visual
        if visual is None:
            continue
        if visual.kind == "texture":
            material = getattr(visual, "material", None)
            if material is not None:
                stats.materials.append(type(material).__name__)
                image = _texture_image(material)
                if image is not None:
                    stats.textured_meshes += 1
                    stats.texture_sizes.append(list(image.size))
        elif visual.kind == "vertex":
            stats.vertex_colour_meshes += 1
    return stats


def _clean_meshes(scene: Any) -> dict[str, int]:
    """Removes junk that does not change the look of the model. Returns what was removed."""
    import trimesh

    removed = {"non_mesh_items": 0, "degenerate_faces": 0, "duplicate_faces": 0, "unused_vertices": 0}
    for name in list(scene.geometry):
        mesh = scene.geometry[name]
        if not isinstance(mesh, trimesh.Trimesh):
            scene.delete_geometry(name)
            removed["non_mesh_items"] += 1
            continue
        if len(mesh.faces) == 0:
            scene.delete_geometry(name)
            removed["non_mesh_items"] += 1
            continue

        # Only touch a mesh when something must go: editing clears trimesh's cache, which
        # would also drop the normals stored in the file.
        keep = mesh.nondegenerate_faces()
        if not keep.all():
            removed["degenerate_faces"] += int((~keep).sum())
            mesh.update_faces(keep)

        unique = mesh.unique_faces()
        if not unique.all():
            removed["duplicate_faces"] += int((~unique).sum())
            mesh.update_faces(unique)

        used = np.zeros(len(mesh.vertices), dtype=bool)
        used[mesh.faces.reshape(-1)] = True
        if not used.all():
            removed["unused_vertices"] += int((~used).sum())
            mesh.remove_unreferenced_vertices()  # also trims the matching UVs / colours

        if len(mesh.faces) == 0:
            scene.delete_geometry(name)
            removed["non_mesh_items"] += 1
    return removed


def _normalise(scene: Any, target_size: float) -> float:
    """Centre on X/Z, put the base on Y=0, scale the largest side to target_size. Returns the scale."""
    bounds = np.asarray(scene.bounds, dtype=np.float64)
    size = float((bounds[1] - bounds[0]).max())
    scale = target_size / size
    centre = (bounds[0] + bounds[1]) / 2.0
    transform = np.eye(4)
    transform[:3, :3] *= scale
    transform[:3, 3] = -np.array([centre[0], bounds[0][1], centre[2]]) * scale
    scene.apply_transform(transform)
    return scale


def _surface_checks(scene: Any) -> tuple[bool, bool, float, float | None]:
    import trimesh

    meshes = [m for m in scene.geometry.values() if isinstance(m, trimesh.Trimesh)]
    watertight = all(m.is_watertight for m in meshes)
    winding = all(m.is_winding_consistent for m in meshes)
    combined = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
    area = float(combined.area)
    volume = float(abs(combined.volume)) if watertight else None
    return watertight, winding, area, volume


def _same_appearance(before: MeshStats, after: MeshStats) -> bool:
    return (
        after.meshes == before.meshes
        and after.faces == before.faces
        and after.textured_meshes == before.textured_meshes
        and after.vertex_colour_meshes == before.vertex_colour_meshes
    )


# ---------------------------------------------------------------- main entry point


def _validate_sync(raw_path: Path, output_path: Path) -> ValidationResult:
    started = time.perf_counter()
    warnings: list[str] = []
    scene = _load_scene(raw_path)
    before = _stats(scene)

    # ---- hard checks
    if before.meshes == 0 or before.vertices < MIN_VERTICES or before.faces < MIN_FACES:
        raise PipelineError(
            f"The 3D model is empty ({before.vertices} vertices, {before.faces} faces). Try again or try a clearer sketch."
        )
    for mesh in scene.geometry.values():
        vertices = np.asarray(getattr(mesh, "vertices", np.zeros((0, 3))))
        if vertices.size and not np.isfinite(vertices).all():
            raise PipelineError("The 3D model contains invalid (NaN or infinite) coordinates.")

    removed = _clean_meshes(scene)
    after_clean = _stats(scene)
    if after_clean.faces < MIN_FACES:
        raise PipelineError("The 3D model has no usable surface after removing broken triangles.")

    bounds = np.asarray(scene.bounds, dtype=np.float64)
    original_extents = bounds[1] - bounds[0]
    largest = float(original_extents.max())
    if not np.isfinite(largest) or largest <= 0:
        raise PipelineError("The 3D model has no size (all points are in the same place).")
    if float(original_extents.min()) / largest < MIN_THICKNESS_RATIO:
        raise PipelineError("The 3D model is completely flat. Try a sketch that shows the object's depth.")

    # ---- soft checks
    watertight, winding, area, volume = _surface_checks(scene)
    if not watertight:
        warnings.append("Mesh is not watertight (normal for single-view models; fine for viewing and most engines).")
    if not winding:
        warnings.append("Some triangles face inconsistent directions; the viewer renders both sides.")
    if after_clean.textured_meshes == 0 and after_clean.vertex_colour_meshes == 0:
        warnings.append("Model has no texture or colours; it will appear in a plain material.")
    if after_clean.faces > LARGE_FACE_COUNT:
        warnings.append(f"Model is heavy ({after_clean.faces:,} triangles) and may load slowly.")

    scale = 1.0
    if settings.mesh_normalize:
        scale = _normalise(scene, settings.mesh_target_size)

    # ---- export and verify nothing visual was lost
    used_original = False
    try:
        output_path.write_bytes(scene.export(file_type="glb", include_normals=True))
        after = _stats(_load_scene(output_path))
        if not _same_appearance(after_clean, after):
            raise ValueError(f"re-export changed the model ({asdict(after_clean)} -> {asdict(after)})")
    except Exception as exc:
        logger.warning("Cleaned GLB rejected, shipping the original: %s", str(exc)[:300])
        shutil.copyfile(raw_path, output_path)
        after, used_original, scale = before, True, 1.0
        warnings.append("Shipped the model exactly as generated (clean-up would have altered its materials).")

    extents = original_extents * scale
    return ValidationResult(
        glb_path=output_path,
        valid=True,
        used_original=used_original,
        before=before,
        after=after,
        watertight=watertight,
        winding_consistent=winding,
        extents=[round(float(x), 4) for x in extents],
        original_extents=[round(float(x), 4) for x in original_extents],
        scale_factor=round(scale, 6),
        removed=removed,
        surface_area=round(area * scale * scale, 5),
        volume=round(volume * scale**3, 6) if volume is not None else None,
        file_size_bytes=output_path.stat().st_size,
        warnings=warnings,
        seconds=round(time.perf_counter() - started, 2),
    )


async def validate_mesh(raw_path: Path, output_path: Path, report_path: Path) -> ValidationResult:
    """Runs Stage 7 in a worker thread and writes validation.json."""
    try:
        import trimesh  # noqa: F401
    except ImportError as exc:
        raise PipelineError("trimesh is not installed. Run: pip install -r requirements.txt") from exc
    try:
        result = await asyncio.to_thread(_validate_sync, raw_path, output_path)
    except PipelineError:
        raise
    except Exception as exc:
        logger.exception("Mesh validation crashed")
        raise PipelineError(f"Mesh validation failed: {str(exc)[:200]}") from exc
    report_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result
