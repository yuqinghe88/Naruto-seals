"""Shared multi-knot controls and tangent sampling; no FK or file output."""
import copy
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation,RotationSpline
import anatomy_authority as anatomy
from compiler.relations import common_fragments
from animation_model.rigs.spring import pose_copy
DIGITS=('index','middle','ring','pinky')

def shared_curve(values,tangent_scale=None):
    values=np.asarray(values,float);tangents=np.zeros_like(values);delta=np.diff(values,axis=0)
    for i in range(1,len(values)-1):
        a,b=delta[i-1],delta[i];same=a*b>0
        tangents[i][same]=2*a[same]*b[same]/(a[same]+b[same])
    if tangent_scale is not None:tangents*=tangent_scale
    return CubicHermiteSpline(np.arange(len(values),dtype=float),values,tangents)


class SequencePose:
    def __init__(self,poses,nodes,edges,assembler,policy):
        self.poses=poses;self.nodes=nodes;self.edges=copy.deepcopy(edges);self.policy=policy;self.end=len(poses)-1
        scale=np.ones((len(poses),2,21));self.windows=[]
        for k in range(1,len(nodes)-1):
            fragments=common_fragments(nodes[k-1:k+2]);digits={t.split('_',1)[1] for f in fragments for t in f['tokens']}
            self.windows.append({'at':nodes[k]['id'],'ids':[n['id'] for n in nodes[k-1:k+2]],
                'shared_fragments':fragments,'shared_digits':sorted(digits),'one_shared_knot':True,
                'region_contact_hold':'unknown','freeze_constraints':[]})
            for side in range(2):
                for digit in DIGITS:
                    if digit not in digits:
                        i=4*DIGITS.index(digit);scale[k,side,i:i+4]=1-policy['independent_digit_phase_lead']*np.pi
                if 'thumb' not in digits:scale[k,side,16:]=1-policy['independent_digit_phase_lead']*np.pi
        self.qcurve=shared_curve([p['q'] for p in poses],scale)
        self.ccurve=shared_curve([p['centers'] for p in poses])
        self.rot=[RotationSpline(np.arange(len(poses),dtype=float),Rotation.from_matrix([p['rotations'][s] for p in poses])) for s in (0,1)]
        self.clearance=[]
        for i,(a,b,edge) in enumerate(zip(poses,poses[1:],edges)):
            mid=self.ccurve(i+.5);v=mid[1]-mid[0]
            if np.linalg.norm(v)<1e-9:v=b['centers'][1]-b['centers'][0]
            if np.linalg.norm(v)<1e-9:raise ValueError('Relative workspace direction unresolved')
            direction=v/np.linalg.norm(v);clear,distance=assembler.separation(a,b)
            amount=max(0.,distance-float(np.linalg.norm(mid[1]-mid[0])))/2
            self.clearance.append({'direction':direction,'separation_delta':amount,'conservative_distance_m':distance})
        self.workspace_overrides={}

    def __call__(self,t):
        if not 0<=t<=self.end:raise ValueError('Sequence time outside registered knots')
        if abs(t-round(t))<1e-12:return pose_copy(self.poses[int(round(t))])
        i=int(np.floor(t));u=t-i
        q=np.array([anatomy.project(x) for x in self.qcurve(t)]);centers=self.ccurve(t).copy()
        rt=2*t*t-t*t*t if t<1 else self.end-(2*(self.end-t)**2-(self.end-t)**3) if t>self.end-1 else t
        rotations=np.array([r(rt).as_matrix() for r in self.rot])
        mode=self.workspace_overrides.get(i,self.edges[i]['mode']);info=self.clearance[i]
        bump=16*u*u*(1-u)*(1-u)  # zero value AND derivative at every seal
        if mode=='release_rebuild':
            offset=bump*info['separation_delta']*info['direction'];centers[0]-=offset;centers[1]+=offset
        elif mode=='effector_clearance':
            side=1-('left','right').index(self.edges[i]['support_hand'])
            centers[side]+=bump*self.policy['effector_clearance_prior_m']*info['direction']*(1 if side else -1)
        elif mode=='compact_fallback':
            offset=bump*self.policy['effector_clearance_prior_m']*info['direction']/2
            centers[0]-=offset;centers[1]+=offset
        return {'q':q,'centers':centers,'rotations':rotations}


def knot_derivatives(path,k):
    e=1e-5;a,b,c=path(k-e),path(float(k)),path(k+e)
    result={'joint_left_speed':float(np.linalg.norm((b['q']-a['q'])/e)),
        'joint_right_speed':float(np.linalg.norm((c['q']-b['q'])/e)),
        'joint_tangent_difference':float(np.linalg.norm((c['q']-2*b['q']+a['q'])/e)),
        'center_tangent_difference':float(np.linalg.norm((c['centers']-2*b['centers']+a['centers'])/e)),
        'wrist':[],'units':'per normalized edge time; visual kinematics only'}
    for side in (0,1):
        left=Rotation.from_matrix(a['rotations'][side].T@b['rotations'][side]).as_rotvec()/e
        right=Rotation.from_matrix(b['rotations'][side].T@c['rotations'][side]).as_rotvec()/e
        result['wrist'].append({'hand':('left','right')[side],'left_speed':float(np.linalg.norm(left)),
            'right_speed':float(np.linalg.norm(right)),'tangent_difference':float(np.linalg.norm(left-right))})
    return result

