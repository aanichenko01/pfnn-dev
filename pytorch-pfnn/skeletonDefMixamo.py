"""
ROOT mixamorig:Hips
JOINT mixamorig:Spine
  JOINT mixamorig:Spine1
    JOINT mixamorig:Spine2
      JOINT mixamorig:Neck
        JOINT mixamorig:Head
          JOINT mixamorig:HeadTop_End
      JOINT mixamorig:RightShoulder
        JOINT mixamorig:RightArm
          JOINT mixamorig:RightForeArm
            JOINT mixamorig:RightHand
              JOINT mixamorig:RightHandThumb1
                JOINT mixamorig:RightHandThumb2
                  JOINT mixamorig:RightHandThumb3
                    JOINT mixamorig:RightHandThumb4
              JOINT mixamorig:RightHandIndex1
                JOINT mixamorig:RightHandIndex2
                  JOINT mixamorig:RightHandIndex3
                    JOINT mixamorig:RightHandIndex4
              JOINT mixamorig:RightHandMiddle1
                JOINT mixamorig:RightHandMiddle2
                  JOINT mixamorig:RightHandMiddle3
                    JOINT mixamorig:RightHandMiddle4
              JOINT mixamorig:RightHandRing1
                JOINT mixamorig:RightHandRing2
                  JOINT mixamorig:RightHandRing3
                    JOINT mixamorig:RightHandRing4
              JOINT mixamorig:RightHandPinky1
                JOINT mixamorig:RightHandPinky2
                  JOINT mixamorig:RightHandPinky3
                    JOINT mixamorig:RightHandPinky4
      JOINT mixamorig:LeftShoulder
        JOINT mixamorig:LeftArm
          JOINT mixamorig:LeftForeArm
             JOINT mixamorig:LeftHand
              JOINT mixamorig:LeftHandThumb1
                JOINT mixamorig:LeftHandThumb2
                  JOINT mixamorig:LeftHandThumb3
                    JOINT mixamorig:LeftHandThumb4
              JOINT mixamorig:LeftHandIndex1
                JOINT mixamorig:LeftHandIndex2
                  JOINT mixamorig:LeftHandIndex3
                    JOINT mixamorig:LeftHandIndex4
              JOINT mixamorig:LeftHandMiddle1
                JOINT mixamorig:LeftHandMiddle2
                  JOINT mixamorig:LeftHandMiddle3
                    JOINT mixamorig:LeftHandMiddle4
              JOINT mixamorig:LeftHandRing1
                JOINT mixamorig:LeftHandRing2
                  JOINT mixamorig:LeftHandRing3
                    JOINT mixamorig:LeftHandRing4
              JOINT mixamorig:LeftHandPinky1
                JOINT mixamorig:LeftHandPinky2
                  JOINT mixamorig:LeftHandPinky3
                    JOINT mixamorig:LeftHandPinky4
JOINT mixamorig:RightUpLeg
  JOINT mixamorig:RightLeg
    JOINT mixamorig:RightFoot
      JOINT mixamorig:RightToeBase
        JOINT mixamorig:RightToe_End
JOINT mixamorig:LeftUpLeg
  JOINT mixamorig:LeftLeg
    JOINT mixamorig:LeftFoot
      JOINT mixamorig:LeftToeBase
        JOINT mixamorig:LeftToe_End
"""


JOINT_NUM = 65

SDR_L, SDR_R, HIP_L, HIP_R = 32, 8, 60, 55
FOOT_L = [63,64]
FOOT_R = [58,59]
HEAD = 6 #if potential problems make it 5?
FILTER_OUT = []
JOINT_SCALE = 5.644

JOINT_WEIGHTS = [
    1,
    1, 1, 1, 
    1, 1, 1e-10, 
    1, 1, 1, 1, 
    1e-10, 1e-10, 1e-10, 1e-10,
    1e-10, 1e-10, 1e-10, 1e-10,
    1e-10, 1e-10, 1e-10, 1e-10,
    1e-10, 1e-10, 1e-10, 1e-10,
    1e-10, 1e-10, 1e-10, 1e-10,
    1, 1, 1, 1,
    1e-10, 1e-10, 1e-10, 1e-10,
    1e-10, 1e-10, 1e-10, 1e-10,
    1e-10, 1e-10, 1e-10, 1e-10,
    1e-10, 1e-10, 1e-10, 1e-10,
    1e-10, 1e-10, 1e-10, 1e-10,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1 ]