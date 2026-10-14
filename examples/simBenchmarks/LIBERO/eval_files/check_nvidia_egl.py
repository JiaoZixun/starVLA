"""Fail early unless robosuite can create a real NVIDIA EGL context."""

import os


if os.environ.get("MUJOCO_GL", "").lower() != "egl":
    raise SystemExit("LIBERO evaluation requires MUJOCO_GL=egl.")
if os.environ.get("PYOPENGL_PLATFORM", "").lower() != "egl":
    raise SystemExit("LIBERO evaluation requires PYOPENGL_PLATFORM=egl.")

try:
    device_id = int(os.environ["MUJOCO_EGL_DEVICE_ID"])
except (KeyError, ValueError) as exc:
    raise SystemExit("Set MUJOCO_EGL_DEVICE_ID to the selected physical GPU index.") from exc

from robosuite.renderers.context.egl_context import EGLGLContext
from OpenGL import GL


context = EGLGLContext(64, 64, device_id=device_id)
try:
    context.make_current()
    vendor = GL.glGetString(GL.GL_VENDOR)
    renderer = GL.glGetString(GL.GL_RENDERER)
    version = GL.glGetString(GL.GL_VERSION)
    vendor = vendor.decode() if vendor else ""
    renderer = renderer.decode() if renderer else ""
    version = version.decode() if version else ""
    if not vendor.startswith("NVIDIA"):
        raise SystemExit(
            f"Expected NVIDIA EGL on GPU {device_id}, got vendor={vendor!r}, renderer={renderer!r}."
        )
    print(
        f"NVIDIA EGL ready: device={device_id}, vendor={vendor}, "
        f"renderer={renderer}, version={version}"
    )
finally:
    context.free()
