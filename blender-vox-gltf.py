bl_info = {
    "name": "VOX → GLB Import & Tracking",
    "author": "Your Name",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "3D View > N-Panel > VOX",
    "description": "Converts .vox using an external script, imports the .glb, and tracks provenance for easy reimport.",
    "category": "Import-Export",
}

import bpy
from bpy.props import (
    StringProperty,
    CollectionProperty,
    PointerProperty,
    FloatProperty,
    BoolProperty,
)
from bpy.types import (
    Operator,
    Panel,
    AddonPreferences,
    PropertyGroup,
)

import os
import sys
import shlex
import subprocess
from pathlib import Path
from datetime import datetime

# -----------------------------------------------------------------------------
# Add-on Preferences
# -----------------------------------------------------------------------------

class VOX2GLB_AddonPreferences(AddonPreferences):
    bl_idname = __name__

    converter_path: StringProperty(
        name="Converter Script/Executable",
        description="Path to your .vox → .glb converter (Python script or executable)",
        subtype='FILE_PATH',
        default="/home/dylan/dev/vox-gltf/target/release/vox_gltf",
    )

    extra_args: StringProperty(
        name="Extra Args",
        description="Extra CLI args passed to the converter (optional)",
        default="",
    )

    use_same_folder: BoolProperty(
        name="Write GLB next to VOX",
        default=True,
        description="If enabled, the .glb will be created next to the source .vox with the same name",
    )

    output_folder: StringProperty(
        name="Output Folder (if not same)",
        description="Folder to write .glb files when 'Write GLB next to VOX' is disabled",
        subtype='DIR_PATH',
        default="",
    )

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.prop(self, "converter_path")
        col.prop(self, "extra_args")
        col.separator()
        col.prop(self, "use_same_folder")
        sub = col.column()
        sub.enabled = not self.use_same_folder
        sub.prop(self, "output_folder")

# -----------------------------------------------------------------------------
# Data Model for Tracking
# -----------------------------------------------------------------------------

class VoxAsset(PropertyGroup):
    vox_path: StringProperty(name="VOX Path")
    glb_path: StringProperty(name="GLB Path")
    root_collection: StringProperty(name="Imported Collection")
    mtime: FloatProperty(name="VOX mtime (epoch)")

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

VOX_COLLECTION_NAME = "VOX Imports"


def ensure_vox_collection():
    coll = bpy.data.collections.get(VOX_COLLECTION_NAME)
    if not coll:
        coll = bpy.data.collections.new(VOX_COLLECTION_NAME)
        bpy.context.scene.collection.children.link(coll)
    return coll


def get_new_objects(before_names):
    return [obj for obj in bpy.data.objects if obj.name not in before_names]


def get_new_collections(before_names):
    return [c for c in bpy.data.collections if c.name not in before_names]


def _collection_root_objects(coll):
    try:
        return [o for o in coll.objects if o.parent is None]
    except Exception:
        return []


def preserve_root_transform(old_coll, new_coll):
    """Try to preserve the root transform from old_coll to new_coll.

    Strategy:
    - If each collection has exactly one root object, copy matrix_world.
    - Otherwise, copy transforms for roots that match by name.
    Returns True on any successful application, False otherwise.
    """
    try:
        old_roots = _collection_root_objects(old_coll) if old_coll else []
        new_roots = _collection_root_objects(new_coll) if new_coll else []
        if not old_roots or not new_roots:
            return False
        if len(old_roots) == 1 and len(new_roots) == 1:
            new_roots[0].matrix_world = old_roots[0].matrix_world.copy()
            return True
        applied = False
        for o in old_roots:
            dst = new_coll.objects.get(o.name)
            if dst is not None:
                dst.matrix_world = o.matrix_world.copy()
                applied = True
        return applied
    except Exception:
        return False


def import_glb(filepath: str):
    # Capture pre-state
    before_obj_names = {o.name for o in bpy.data.objects}
    before_coll_names = {c.name for c in bpy.data.collections}

    # Import via built-in glTF importer
    result = bpy.ops.import_scene.gltf(filepath=filepath)
    if 'FINISHED' not in result:
        raise RuntimeError(f"Failed to import GLB: {filepath}")

    new_objs = get_new_objects(before_obj_names)
    new_colls = get_new_collections(before_coll_names)

    # The glTF importer usually creates a new top-level collection; pick the newest
    imported_coll = None
    if new_colls:
        # Heuristic: use the last created collection
        imported_coll = new_colls[-1]
    else:
        # Fallback: gather a temporary collection from new objs
        imported_coll = bpy.data.collections.new(f"GLB_{Path(filepath).stem}")
        bpy.context.scene.collection.children.link(imported_coll)
        for o in new_objs:
            if not any(imported_coll.objects.get(o.name) for _ in [0]):
                imported_coll.objects.link(o)

    # Move/ensure all imported objects exist under VOX_COLLECTION_NAME as parent container
    vox_container = ensure_vox_collection()
    # Ensure linkage by name (bpy_prop_collection contains() expects strings)
    if vox_container.children.get(imported_coll.name) is None:
        # Link imported_coll under the VOX container if not already hierarchically linked
        try:
            vox_container.children.link(imported_coll)
        except RuntimeError:
            # Already linked somewhere; ignore
            pass

    return imported_coll


def run_converter(converter_path: str, vox_path: str, out_glb: str, extra_args: str = ""):
    if not converter_path:
        raise RuntimeError("Converter path not set. Go to Preferences > Add-ons > VOX → GLB Import & Tracking.")

    cmd = [converter_path, vox_path, '-d', out_glb]

    if extra_args.strip():
        cmd.extend(shlex.split(extra_args))

    # Ensure output dir exists
    Path(out_glb).parent.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Converter failed (code {}):\nSTDOUT:\n{}\nSTDERR:\n{}".format(
                completed.returncode, completed.stdout, completed.stderr
            )
        )

    if not Path(out_glb).exists():
        raise RuntimeError(f"Converter reported success but GLB not found: {out_glb}")

    return completed.stdout


# -----------------------------------------------------------------------------
# Operators
# -----------------------------------------------------------------------------

class VOX2GLB_OT_import(Operator):
    bl_idname = "vox2glb.import"
    bl_label = "Import .vox (convert to .glb)"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: StringProperty(subtype='FILE_PATH')

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        prefs: VOX2GLB_AddonPreferences = context.preferences.addons[__name__].preferences

        vox_path = self.filepath
        if not vox_path or not vox_path.lower().endswith('.vox'):
            self.report({'ERROR'}, "Please choose a .vox file")
            return {'CANCELLED'}

        vox = Path(vox_path)
        if prefs.use_same_folder:
            out_glb = vox.with_suffix('.glb')
        else:
            if not prefs.output_folder:
                self.report({'ERROR'}, "Output folder not set in preferences")
                return {'CANCELLED'}
            out_glb = Path(prefs.output_folder) / f"{vox.stem}.glb"

        try:
            run_converter(prefs.converter_path, str(vox), str(out_glb), prefs.extra_args)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        try:
            imported_coll = import_glb(str(out_glb))
        except Exception as e:
            self.report({'ERROR'}, f"Import failed: {e}")
            return {'CANCELLED'}

        # Track asset in scene collection property
        sa = context.scene.vox_assets.add()
        sa.vox_path = str(vox)
        sa.glb_path = str(out_glb)
        sa.root_collection = imported_coll.name
        try:
            sa.mtime = os.path.getmtime(str(vox))
        except Exception:
            sa.mtime = 0.0

        self.report({'INFO'}, f"Imported {vox.name} → {imported_coll.name}")
        return {'FINISHED'}


class VOX2GLB_OT_reimport_dirty(Operator):
    bl_idname = "vox2glb.reimport_dirty"
    bl_label = "Reimport Modified VOX"
    bl_description = "Re-run conversion for VOX files that changed on disk and refresh the scene"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        prefs: VOX2GLB_AddonPreferences = context.preferences.addons[__name__].preferences
        scene = context.scene
        changed = 0

        for item in scene.vox_assets:
            vox_p = Path(item.vox_path)
            glb_p = Path(item.glb_path)
            if not vox_p.exists():
                continue
            current_mtime = os.path.getmtime(str(vox_p))
            if current_mtime <= item.mtime:
                continue

            # Reconvert
            try:
                run_converter(prefs.converter_path, str(vox_p), str(glb_p), prefs.extra_args)
            except Exception as e:
                self.report({'ERROR'}, f"Converter failed for {vox_p.name}: {e}")
                continue

            # Import first so we can preserve transforms from the old root
            old_coll = bpy.data.collections.get(item.root_collection)
            try:
                imported_coll = import_glb(str(glb_p))
            except Exception as e:
                self.report({'ERROR'}, f"Reimport failed for {vox_p.name}: {e}")
                continue

            # Try to preserve root transform from old to new
            try:
                preserve_root_transform(old_coll, imported_coll)
            except Exception:
                pass

            # Now remove old imported collection if present
            coll = old_coll
            if coll:
                # Unlink from any parent collections or scene master collections, then delete
                try:
                    # Unlink from parent collections
                    for parent in bpy.data.collections:
                        if parent.children.get(coll.name) is not None:
                            try:
                                parent.children.unlink(coll)
                            except Exception:
                                pass
                    # Unlink from scenes' master collections (defensive)
                    for scn in bpy.data.scenes:
                        if scn.collection.children.get(coll.name) is not None:
                            try:
                                scn.collection.children.unlink(coll)
                            except Exception:
                                pass
                except Exception:
                    pass
                # Remove objects inside
                for obj in list(coll.objects):
                    try:
                        bpy.data.objects.remove(obj, do_unlink=True)
                    except Exception:
                        pass
                try:
                    bpy.data.collections.remove(coll)
                except Exception:
                    pass

            # Update tracking
            item.root_collection = imported_coll.name
            item.mtime = current_mtime
            changed += 1

        self.report({'INFO'}, f"Reimported {changed} modified VOX file(s)")
        return {'FINISHED'}


class VOX2GLB_OT_forget_selected(Operator):
    bl_idname = "vox2glb.forget_selected"
    bl_label = "Forget Selected Import"
    bl_description = "Remove tracking for the VOX asset associated with the active object/collection. (Does not delete objects.)"

    def execute(self, context):
        scene = context.scene
        active_coll_name = None
        if context.view_layer.active_layer_collection:
            active_coll_name = context.view_layer.active_layer_collection.collection.name

        if not active_coll_name:
            self.report({'ERROR'}, "No active collection to forget")
            return {'CANCELLED'}

        removed = 0
        for i in reversed(range(len(scene.vox_assets))):
            if scene.vox_assets[i].root_collection == active_coll_name:
                scene.vox_assets.remove(i)
                removed += 1

        if removed == 0:
            self.report({'WARNING'}, "Active collection is not tracked")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Forgot tracking for {removed} item(s)")
        return {'FINISHED'}


# -----------------------------------------------------------------------------
# UI Panel
# -----------------------------------------------------------------------------

class VOX2GLB_PT_panel(Panel):
    bl_label = "VOX Import"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'VOX'

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.operator(VOX2GLB_OT_import.bl_idname, icon='IMPORT')
        col.operator(VOX2GLB_OT_reimport_dirty.bl_idname, icon='FILE_REFRESH')
        col.separator()

        scene = context.scene
        if len(scene.vox_assets) == 0:
            col.label(text="No tracked VOX imports yet.")
        else:
            box = col.box()
            box.label(text="Tracked Imports:")
            for item in scene.vox_assets:
                row = box.row()
                row.label(text=f"{Path(item.vox_path).name}")
                sub = row.row()
                sub.alignment = 'RIGHT'
                sub.label(text=f"→ {item.root_collection}")
            col.separator()
            col.operator(VOX2GLB_OT_forget_selected.bl_idname, icon='X')

# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

classes = (
    VOX2GLB_AddonPreferences,
    VoxAsset,
    VOX2GLB_OT_import,
    VOX2GLB_OT_reimport_dirty,
    VOX2GLB_OT_forget_selected,
    VOX2GLB_PT_panel,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.vox_assets = CollectionProperty(type=VoxAsset)


def unregister():
    del bpy.types.Scene.vox_assets
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
