"""Registered visual motif interpolation; no target solve or acceptance."""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
import anatomy_authority as anatomy
from animation_model.rigs.spring import pose_copy
from trajectory.timing import progress
SIDES=('left','right')

def rotate(a,b,s):
    return Slerp([0.,1.],Rotation.from_matrix([a,b]))([s]).as_matrix()[0]


def motif_pose(a,b,candidate,t,clear):
    """Different component schedules; no intermediate stop of all components."""
    if t==0:return pose_copy(a)
    if t==1:return pose_copy(b)
    motif=candidate['motif_id'];q=a['q'].copy();rot=[]
    if motif=='preserve_common':
        centers=(1-progress(t,(0.,1.)))*a['centers']+progress(t,(0.,1.))*b['centers']
        windows=[(.06,.95),(.06,.95)]
        rot=[rotate(x,y,progress(t,(.03,.97))) for x,y in zip(a['rotations'],b['rotations'])]
    elif motif=='support_hand':
        support=SIDES.index(candidate['support_hand']);active=1-support
        g=progress(t,(.62,1.));u=progress(t,(.04,.84))
        # Align the target support wrist to the source with a proper rigid
        # transform; move the effector relative to it, then settle the pair.
        align=a['rotations'][support]@b['rotations'][support].T
        aligned=(b['centers']-b['centers'][support])@align.T+a['centers'][support]
        centers=a['centers'].copy()
        centers[active]=(1-u)*a['centers'][active]+u*aligned[active]
        centers[active]+=4*u*(1-u)*(clear[active]-a['centers'][active])
        # Undo target alignment continuously in the final stage, rather than
        # lerping rotations independently of the hand-pair translation.
        undo=rotate(np.eye(3),align.T,g)
        centers=(centers-a['centers'][support])@undo.T+((1-g)*a['centers'][support]+g*b['centers'][support])
        for side in range(2):
            base=(a['rotations'][side] if side==support else
                  rotate(a['rotations'][side],align@b['rotations'][side],u))
            rot.append(undo@base)
        windows=[(.51,.97) if side==support else (.12,.84) for side in range(2)]
    else:
        out=progress(t,(0.,.34));inside=progress(t,(.57,1.))
        centers=a['centers']+out*(clear-a['centers'])+inside*(b['centers']-clear)
        windows=[(.17,.87),(.21,.91)]
        rot=[rotate(x,y,progress(t,(.12,.88),.02*side)) for side,(x,y) in enumerate(zip(a['rotations'],b['rotations']))]
    for side in range(2):
        start,end=windows[side]
        for digit in range(4):
            for j in (0,1,2):
                lag=.025 if j==2 else 0.;s=progress(t,(start+lag,end+lag))
                i=4*digit+j;q[side,i]=(1-s)*a['q'][side,i]+s*b['q'][side,i]
        for i in (16,17,19,20):
            s=progress(t,(start,end));q[side,i]=(1-s)*a['q'][side,i]+s*b['q'][side,i]
        q[side]=anatomy.project(q[side])
    return {'q':q,'rotations':np.array(rot),'centers':centers}

