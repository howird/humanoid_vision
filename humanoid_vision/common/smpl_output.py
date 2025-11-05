from dataclasses import dataclass

from humanoid_vision.common.types import Betas, BodyPose, GlobalOrient


@dataclass
class HMRSMPLOutput:
    """SMPL prediction parameters.

    Attributes:
        global_orient: Global rotation matrices of shape (B, 1, 3, 3)
        body_pose: Body joint rotation matrices of shape (B, 23, 3, 3)
        betas: Shape parameters of shape (B, 10)
    """

    global_orient: GlobalOrient
    body_pose: BodyPose
    betas: Betas
