import torch
import torch.nn.functional as F

import numpy as np
import torch.nn as nn
import neural_renderer as nr
from jaxtyping import Float, Int, Bool, jaxtyped
from beartype import beartype

from humanoid_vision.utils.pylogger_phalp import get_pylogger
from humanoid_vision.configs.base import CACHE_DIR


log = get_pylogger(__name__)


@jaxtyped(typechecker=beartype)
def unproject_uvmap_to_mesh(
    bary_map: Float[torch.Tensor, "256 256 3"],
    face_map: Int[torch.Tensor, "256 256"],
    verts: Float[torch.Tensor, "batch num_verts 3"],
    faces: Int[torch.Tensor, "num_faces 3"],
) -> tuple[Float[torch.Tensor, "batch _ 3"], Bool[torch.Tensor, "256 256"]]:
    valid_mask = face_map >= 0

    fmap_flat = face_map[valid_mask]  # N
    bmap_flat = bary_map[valid_mask, :]  # N,3

    face_vids = faces[fmap_flat, :]  # N,3
    face_verts = verts[:, face_vids, :]  # B,N,3,3

    bs = face_verts.shape
    map_verts = torch.einsum("bnij,ni->bnj", face_verts, bmap_flat)  # B,N,3

    return map_verts, valid_mask


class UVTextureRenderer(nn.Module):
    """Renderer responsible for projecting image evidence to the UV atlas."""

    def __init__(self):
        super().__init__()

        self.img_size = 256
        self.focal_length = 5000.0

        bmap_path = CACHE_DIR / "phalp/3D/bmap_256.npy"
        fmap_path = CACHE_DIR / "phalp/3D/fmap_256.npy"
        bary_map = np.load(bmap_path)
        face_map = np.load(fmap_path)
        self.register_buffer("tex_bmap", torch.tensor(bary_map, dtype=torch.float))
        self.register_buffer("tex_fmap", torch.tensor(face_map, dtype=torch.long))

        self.renderer = nr.Renderer(
            dist_coeffs=None,
            orig_size=self.img_size,
            image_size=self.img_size,
            light_intensity_ambient=1,
            light_intensity_directional=0,
            anti_aliasing=False,
        )

    @jaxtyped(typechecker=beartype)
    def render_uv_image(
        self,
        pred_vertices: Float[torch.Tensor, "batch verts 3"],
        pred_cam_t: Float[torch.Tensor, "batch 3"],
        faces: Int[torch.Tensor, "faces 3"],
        image: Float[torch.Tensor, "batch 3 256 256"],
        mask: Float[torch.Tensor, "batch 256 256"],
    ) -> Float[torch.Tensor, "batch 4 256 256"]:
        """
        Args:
            pred_vertices: predicted 3D mesh
            pred_cam_t: camera translation
            faces: mesh faces
            image: original 2D image
        Returns: uv texture map
        """
        device = pred_vertices.device
        # Align SMPL mesh with camera translation.
        pred_verts = pred_vertices + pred_cam_t.unsqueeze(1)

        # uses pre-defined canonical UV map definitions (`tex_bmap`, `tex_fmap`)
        # to "unproject" the 2D UV atlas coordinates onto the surface of the 3D mesh
        # performs a standard perspective projection on these 3D coordinates (`map_verts`)
        # to find where they land on the 2D image plane
        map_verts, valid_mask = unproject_uvmap_to_mesh(
            self.tex_bmap,
            self.tex_fmap,
            pred_verts,
            faces,
        )

        focal = self.focal_length / (self.img_size / 2)
        # map_verts_proj: texture flow, the set of (x, y) coords, which tell the renderer
        #   which pixel in the 2D image corresponds to each pixel in the 2D texture map.
        map_verts_proj = focal * map_verts[:, :, :2] / map_verts[:, :, 2:3]
        map_verts_depth = map_verts[:, :, 2]

        # Camera intrinsics/extrinsics for depth rendering.
        K = torch.eye(3, device=device)
        K[0, 0] = K[1, 1] = self.focal_length
        K[0, 2] = K[1, 2] = self.img_size / 2
        K = K.unsqueeze(0)

        R = torch.eye(3, device=device).unsqueeze(0)
        t = torch.zeros(3, device=device).unsqueeze(0)

        # Any part of the texture map corresponding to a 3D point that is
        # occluded (hidden) by another part of the mesh will be masked out.
        # renders a depth map (`rend_depth`) of the entire 3D mesh
        rend_depth = self.renderer(
            pred_verts,
            faces[None].expand(pred_verts.shape[0], -1, -1).int(),
            mode="depth",
            K=K,
            R=R,
            t=t,
        )

        rend_depth_at_proj = (
            F.grid_sample(
                rend_depth[:, None, :, :],
                map_verts_proj[:, None, :, :],
            )
            .squeeze(1)
            .squeeze(1)
        )

        # bilinear sampling the input 2D image at the calculated coordinates
        img_rgba = torch.cat([image, mask[:, None, :, :]], dim=1)
        img_rgba_at_proj = F.grid_sample(
            img_rgba,
            map_verts_proj[:, None, :, :],
        ).squeeze(2)

        # checks if the 3D points from the unprojected UV map (`map_verts_depth`)
        # are actually visible (i.e., in front of) the rendered mesh surface.
        visibility_mask = map_verts_depth <= (rend_depth_at_proj + 1e-4)
        img_rgba_at_proj[:, 3, :][~visibility_mask] = 0

        uv_image = torch.zeros(
            (image.shape[0], 4, self.img_size, self.img_size),
            dtype=torch.float,
            device=device,
        )
        uv_image[:, :, valid_mask] = img_rgba_at_proj

        return uv_image
