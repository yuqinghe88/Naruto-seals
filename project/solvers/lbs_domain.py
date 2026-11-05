"""Current LBS state, independent control map and measured predicates.

Display skin/radii are never promoted to contact or collision geometry.
"""
from __future__ import annotations
import copy
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation
import anatomy_authority as anatomy
from nih_lbs_rig import load_rig,pose_hand,result_json,apply_action,apply_placement_action,placement_for

SIDES=('left','right')
JOINT_NAMES=[]
for side in SIDES:
 for digit in anatomy.DIGITS:
  JOINT_NAMES += [f'{side}.{digit}.{k}' for k in anatomy.KEYS]
 JOINT_NAMES += [f'{side}.thumb.{k}' for k in anatomy.THUMB_KEYS]
GLOBAL_NAMES=[f'{side}.global.t{axis}' for side in SIDES for axis in 'xyz']+[f'{side}.wrist.r{axis}' for side in SIDES for axis in 'xyz']
NAMES=JOINT_NAMES+GLOBAL_NAMES
INDEX={name:i for i,name in enumerate(NAMES)}
DEPENDENT={side*21+i for side in range(2) for i in (3,7,11,15,18)}
INDEPENDENT=[n for i,n in enumerate(NAMES) if i not in DEPENDENT]

class ContractError(ValueError):pass

class LBSState:
 def __init__(self,result):
  if result.get('kinematic_authority')!='nih_lbs':raise ContractError('LBS initial state required; legacy FK cannot be silently substituted')
  if result.get('anatomy_authority',{}).get('sha256')!=anatomy.SHA256:raise ContractError('stale anatomy authority; explicitly remap initial state first')
  self.rig=load_rig();self.base={};self.x=np.zeros(54);self.history=[];self.established={}
  for i,side in enumerate(SIDES):
   hand=next(h for h in result['hands'] if h['side']==side);q=np.asarray(hand['control_q_rad'],float)
   if anatomy.check(q)['status']!='pass':raise ContractError('initial controls fail anatomy authority; native bind is not implicitly a valid command')
   M=np.asarray(hand['lbs_binding']['bone_matrices']['wrist-palm'],float);R=M[:3,:3];t=M[:3,3]
   rebuilt=pose_hand(side,q,{'rotation':R.tolist(),'translation_m':t.tolist()},self.rig)
   for digit,f in hand['fingers'].items():
    if not np.allclose(f['points_world_m'],rebuilt['fingers'][digit]['points_world_m'],atol=1e-10,rtol=0):raise ContractError('initial stored points inconsistent with current LBS FK')
   self.x[21*i:21*i+21]=q
   self.base[side]={'rotation':R.copy(),'wrist_world':R@np.asarray(self.rig['hands'][side]['wrist_m'])+t}
  self.initial=self.x.copy()
 def snapshot(self):return {'x':self.x.copy(),'history':copy.deepcopy(self.history),'established':copy.deepcopy(self.established)}
 def restore(self,s):self.x=s['x'].copy();self.history=copy.deepcopy(s['history']);self.established=copy.deepcopy(s['established'])
 def fingerprint(self):return hashlib.sha256(json.dumps({'x':self.x.tolist(),'history':self.history,'established':self.established},sort_keys=True,allow_nan=False).encode()).hexdigest()
 def project(self,x,active):
  out=np.asarray(x,float).copy()
  for i,side in enumerate(SIDES):
   ids=[INDEX[n]-21*i for n in active if n.startswith(side+'.') and INDEX[n]<42]
   if ids:out[i*21:i*21+21]=anatomy.project(out[i*21:i*21+21],ids)
  return out
 def fk(self,x):
  hands=[]
  for i,side in enumerate(SIDES):
   R=Rotation.from_rotvec(x[48+3*i:51+3*i]).as_matrix()@self.base[side]['rotation']
   pivot=self.base[side]['wrist_world']+x[42+3*i:45+3*i];t=pivot-R@np.asarray(self.rig['hands'][side]['wrist_m'])
   hands.append(pose_hand(side,x[21*i:21*i+21],{'rotation':R.tolist(),'translation_m':t.tolist()},self.rig))
  return result_json(hands,[x[:21].tolist(),x[21:42].tolist()])
 def action_target(self,action):
  target=self.x.copy();family=action['family'];p=action.get('parameters',{})
  if set(action)-{'family','parameters','effects'}:raise ContractError('action guards/extra fields require an explicit runtime binding')
  if family=='Control':
   for name,value in p.get('set',{}).items():
    if name not in INDEX:raise ContractError('unknown control '+name)
    i=INDEX[name]
    if i in DEPENDENT:raise ContractError('dependent DOF is not an independent command: '+name)
    if not np.isfinite(value):raise ContractError('finite control required')
    if i<42 and not anatomy.LIMITS_RAD[i%21,0]<=value<=anatomy.LIMITS_RAD[i%21,1]:raise ContractError('explicit control outside authority: '+name)
    target[i]=float(value)
   return self.project(target,list(p.get('set',{})))
  side=p.get('hand')
  if side not in SIDES:raise ContractError('explicit hand required for numeric action')
  i=SIDES.index(side)
  if family in ('Flex','Spread','Thumb'):
   target[21*i:21*i+21]=apply_action(target[21*i:21*i+21],action);return target
  if family in ('Palm','RelHand'):
   R=Rotation.from_rotvec(self.x[48+3*i:51+3*i]).as_matrix()@self.base[side]['rotation'];pivot=self.base[side]['wrist_world']+self.x[42+3*i:45+3*i];placement={'rotation':R.tolist(),'translation_m':(pivot-R@np.asarray(self.rig['hands'][side]['wrist_m'])).tolist()}
   moved=apply_placement_action(side,placement,action,self.rig);R=np.asarray(moved['rotation']);t=np.asarray(moved['translation_m'])
   target[42+3*i:45+3*i]=R@np.asarray(self.rig['hands'][side]['wrist_m'])+t-self.base[side]['wrist_world']
   target[48+3*i:51+3*i]=Rotation.from_matrix(R@self.base[side]['rotation'].T).as_rotvec();return target
  raise ContractError('action binding unavailable: '+family)


def anchor(result,spec):
 if not isinstance(spec,dict) or spec.get('hand') not in SIDES:raise ContractError('anchor hand missing')
 h=next(h for h in result['hands'] if h['side']==spec['hand']);structure=spec.get('structure');region=spec.get('region',{})
 if structure=='wrist' and region.get('type')=='joint':return np.asarray(h['wrist_world_m'][1])
 if structure=='palm' and region.get('type')=='joint' and region.get('id')=='rig_center':return np.asarray(h['palm_center_world_m'])
 f=h['fingers'].get(structure)
 if f is None:raise ContractError('anchor structure unresolved')
 points=np.asarray(f['points_world_m'])
 if region.get('type')=='tip':return points[-1]
 if region.get('type')=='joint':
  ids={'MCP':0,'PIP':1,'DIP':2} if structure!='thumb' else {'CMC':0,'MCP':1,'IP':2}
  if region.get('id') in ids:return points[ids[region['id']]]
 if region.get('type')=='segment':
  ids={'segment_1':0,'segment_2':1,'segment_3':2} if structure!='thumb' else {'segment_1':1,'segment_2':2}
  j=ids.get(region.get('id'));u=spec.get('parameterization',{}).get('u')
  if j is not None and isinstance(u,(int,float)) and not isinstance(u,bool) and 0<=u<=1:return (1-u)*points[j]+u*points[j+1]
  raise ContractError('segment point requires explicit u; an interval is not a fixed point')
 raise ContractError('surface or anatomical region not bound to authoritative LBS geometry')


def observe(state,result,predicate):
 """Return actual evidence and normalized objective; unsupported stays unknown."""
 p=predicate;kind=p.get('kind');e={};r=[]
 try:
  fields={'joint':{'kind','dof','range','tolerance'},'point_position':{'kind','anchor','target_m','tolerance_m','frame','frame_hand'},'point_separation':{'kind','a','b','range_m','tolerance_m'},'palm_orientation':{'kind','hand','rotation_matrix','tolerance_rad','frame'},'relation':{'kind','contract'},'physical':{'kind'}}
  unsupported=set(p)-fields.get(kind,set())-{'id','source'}
  if unsupported:raise ContractError('unsupported predicate fields: '+str(sorted(unsupported)))
  if kind=='joint':
   name=p['dof'];i=INDEX[name];v=float(state.x[i]);lo,hi=map(float,p['range']);tol=float(p.get('tolerance',0.))
   if i>=42 or lo>hi or tol<0:raise ContractError('joint range invalid')
   delta=min(v-lo,0.)+max(v-hi,0.);scale=max(tol,1e-6)
   e={'dof':name,'value_rad':v,'range_rad':[lo,hi],'tolerance_rad':tol};r=[delta/scale];passed=abs(delta)<=tol+1e-12
  elif kind=='point_position':
   point=anchor(result,p['anchor']);target=np.asarray(p['target_m'],float);tol=float(p['tolerance_m'])
   if target.shape!=(3,) or not np.isfinite(target).all() or tol<=0:raise ContractError('point target/tolerance invalid')
   frame=p.get('frame');delta=point-target
   if frame=='palm_local':
    h=next(h for h in result['hands'] if h['side']==p['frame_hand']);F=np.asarray(h['palm_rotation_matrix']);local=F.T@(point-np.asarray(h['palm_center_world_m']));delta=local-target
   elif frame!='world':raise ContractError('point target requires explicit frame')
   r=(delta/tol).tolist();passed=np.linalg.norm(delta)<=tol+1e-12;e={'anchor':p['anchor'],'point_world_m':point.tolist(),'error_m':float(np.linalg.norm(delta)),'tolerance_m':tol,'frame':frame}
  elif kind=='point_separation':
   a,b=anchor(result,p['a']),anchor(result,p['b']);distance=float(np.linalg.norm(a-b));lo,hi=map(float,p['range_m']);tol=float(p['tolerance_m'])
   if lo>hi or lo<0 or tol<=0:raise ContractError('distance band invalid')
   delta=min(distance-lo,0)+max(distance-hi,0);r=[delta/tol];passed=abs(delta)<=tol+1e-12;e={'distance_m':distance,'range_m':[lo,hi],'tolerance_m':tol,'scope':'joint/axis point distance; not surface contact'}
  elif kind=='palm_orientation':
   hand=next(h for h in result['hands'] if h['side']==p['hand']);target=np.asarray(p['rotation_matrix'],float);tol=float(p['tolerance_rad'])
   if p.get('frame')!='world' or target.shape!=(3,3) or not np.allclose(target.T@target,np.eye(3),atol=1e-9) or not np.isclose(np.linalg.det(target),1) or tol<=0:raise ContractError('palm orientation frame/target invalid')
   error=Rotation.from_matrix(np.asarray(hand['palm_rotation_matrix'])@target.T).as_rotvec();r=(error/tol).tolist();passed=np.linalg.norm(error)<=tol+1e-12;e={'angle_error_rad':float(np.linalg.norm(error)),'tolerance_rad':tol,'frame':'world','scope':'rigid palm orientation prior; no clinical wrist axis calibration'}
  elif kind=='relation':
   contract=p['contract'];return {'status':'unknown','reason':'LBS_relation_surface_binding_missing','kind':kind,'requested':copy.deepcopy(contract),'missing':['measured or explicitly authorized rig surfaces and predicate-specific binding'],'residual':None,'scope':'legacy relation checker is registered but cannot evaluate the new LBS rig'}
  elif kind=='physical':return {'status':'unknown','reason':'LBS_physics_backend_not_bound','residual':None}
  else:raise ContractError('predicate not registered: '+str(kind))
  if not np.isfinite(r).all():raise ContractError('nonfinite predicate')
  return {'status':'pass' if passed else 'fail','kind':kind,'evidence':e,'residual':r,'scope':'current LBS FK; no collision or continuous-path certificate'}
 except (KeyError,ValueError,TypeError,StopIteration,IndexError) as exc:return {'status':'unknown','kind':kind,'reason':str(exc),'residual':None,'requested':copy.deepcopy(p)}


def controls_for(stage):
 allowed=stage.get('allowed_dofs')
 if not isinstance(allowed,list) or len(set(allowed))!=len(allowed) or any(n not in INDEPENDENT for n in allowed):raise ContractError('allowed_dofs must explicitly list independent registered controls')
 frozen=stage.get('preserve',{}).get('freeze_dofs',[])
 if not isinstance(frozen,list):raise ContractError('freeze_dofs must be a list')
 if any(n not in INDEX for n in frozen):raise ContractError('unknown frozen DOF')
 # A frozen dependent value restricts its driving DOFs too. Never clamp a held
 # DIP/CMC ps independently and violate the authority coupling.
 blocked=set(frozen)
 for name in frozen:
  if name.endswith('.dip'):blocked.add(name[:-3]+'pip')
  if name.endswith('.cmc_ps'):blocked.update([name[:-6]+'cmc_fe',name[:-6]+'cmc_aa'])
 return [n for n in allowed if n not in blocked],sorted(blocked)
