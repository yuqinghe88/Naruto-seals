"""Visual timing and coupled controls, independent of actor/rendering."""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
import anatomy_authority as anatomy

def progress(t,window,delay=0.):
    a,b=window
    if not 0<=a+delay<b+delay<=1:
        raise ValueError('phase windows must finish within the clip')
    s=np.clip((t-a-delay)/(b-a),0.,1.)
    return float(s*s*s*(10.+s*(-15.+6.*s)))


def phase_pose(keys,t,policy):
    if not 0<=t<=1:
        raise ValueError('normalized clip time must be in [0,1]')
    a,b=keys[0]['pose'],keys[-1]['pose']
    if t==0:return {k:v.copy() for k,v in a.items()}
    if t==1:return {k:v.copy() for k,v in b.items()}
    cfg=policy['timing'];clear=keys[1]['pose']['centers']
    out=progress(t,cfg['translation_out']);inside=progress(t,cfg['translation_in'])
    centers=a['centers']+out*(clear-a['centers'])+inside*(b['centers']-clear)
    q=a['q'].copy();rot=[]
    for side in range(2):
        delay=cfg['right_hand_phase_delay'] if side else 0.
        for digit in range(4):
            for offset,window in [(0,'MCP_and_spread'),(1,'MCP_and_spread'),(2,'PIP')]:
                i=4*digit+offset;s=progress(t,cfg[window],delay)
                q[side,i]=(1-s)*a['q'][side,i]+s*b['q'][side,i]
        for i,window in [(16,'thumb_CMC'),(17,'thumb_CMC'),(19,'thumb_MCP_IP'),(20,'thumb_MCP_IP')]:
            s=progress(t,cfg[window],delay)
            q[side,i]=(1-s)*a['q'][side,i]+s*b['q'][side,i]
        q[side]=anatomy.project(q[side])
        s=progress(t,cfg['palm_rotation'],delay)
        rot.append(Slerp([0.,1.],Rotation.from_matrix([a['rotations'][side],b['rotations'][side]]))([s]).as_matrix()[0])
    return {'q':q,'centers':centers,'rotations':np.array(rot)}

