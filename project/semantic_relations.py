"""P0 relation observations based on gesture_feedback's current FK and sensors.

Observations distinguish evidence from topological claims. Unsupported
relations are unknown; no new solver or per-seal limits are introduced.
"""
import copy
import numpy as np
import naruto
from animation_model.geometry import segment_closest
from fist_ring_geometry import measure as measure_fist_rings


def result(status,reason,evidence=None):
 return {'status':status,'reason':reason,'evidence':evidence or {},'source':'gesture_feedback current FK and sensors'}

def _finger(c,anchor):
 if not isinstance(anchor,dict):return None
 side={'left':0,'right':1}.get(anchor.get('hand'));name=anchor.get('structure')
 return next((f for f in c.fingers if f['side']==side and f['name']==name),None)

def _segment(c,anchor):
 f=_finger(c,anchor);reg=anchor.get('region',{}) if isinstance(anchor,dict) else {}
 if not f or reg.get('type')!='segment':return None
 ids={'segment_1':0,'segment_2':1,'segment_3':2};n=ids.get(reg.get('id'))
 if n is None or (f['name']=='thumb' and n==0 and c.model['hands'][f['side']]['fingers']['thumb'].get('embedded_base')):return None
 u=anchor.get('parameterization',{}).get('u_range',[0.,1.])
 if not isinstance(u,list) or len(u)!=2 or not all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in u) or not (0<=u[0]<=u[1]<=1):return None
 a,b=c.p[f['ids'][n:n+2]];return a+(b-a)*u[0],a+(b-a)*u[1],f['radius']

def contact(c,a,b):
 """tip-tip or specified visible segment pair; closest capsule surface gap."""
 fa,fb=_finger(c,a),_finger(c,b)
 if not fa or not fb:return result('unknown','anchor_not_resolved')
 ra,rb=a.get('region',{}),b.get('region',{})
 if ra.get('type')=='tip' and rb.get('type')=='tip':
  gap=float(np.linalg.norm(c.p[fa['ids'][-1]]-c.p[fb['ids'][-1]])-fa['radius']-fb['radius']);scope='tip spheres'
 elif ra.get('type') in ('tip','segment') and rb.get('type') in ('tip','segment'):
  def geometry(anchor,f,region):
   if region.get('type')=='tip':
    p=c.p[f['ids'][-1]];return p,p,f['radius']
   return _segment(c,anchor)
  sa,sb=geometry(a,fa,ra),geometry(b,fb,rb)
  if sa is None or sb is None:return result('unknown','segment_not_resolved')
  pa,pb,_,_=segment_closest(sa[0],sa[1],sb[0],sb[1]);gap=float(np.linalg.norm(pa-pb)-sa[2]-sb[2]);scope='specified capsule segments or tip spheres'
 else:return result('unknown','anchor_pair_not_supported')
 tol=c.cfg['contact_tolerance_m'];pen=c.cfg['penetration_tolerance_m'];return result('pass' if -pen<=gap<=tol else 'fail','contact_band',{'signed_gap_m':gap,'contact_tolerance_m':tol,'penetration_tolerance_m':pen,'approximation':scope,'anchors':[a,b]})

def interlock(c):
 """Report independent observable checks; never equate projected order to topology."""
 rel=c.seal.get('interlock') or {}
 if not rel:return result('unknown','interlock_not_specified')
 o=c._visible_order();g=c._interlock_gaps().tolist();land=c.interlock_landing_margins().tolist();tol=c.cfg['contact_tolerance_m'];pen=c.cfg['penetration_tolerance_m']
 order='unknown' if not o.get('expected') or o.get('missing') else ('pass' if o['pass'] else 'fail')
 gaps='unknown' if not g else ('pass' if all(-pen<=v<=tol for v in g) else 'fail')
 landing='unknown' if not land else ('pass' if min(land)>=-1e-9 else 'fail')
 anchor_observations=[]
 for pair in rel.get('anchor_pairs',[]):
  if isinstance(pair,dict) and isinstance(pair.get('a'),dict) and isinstance(pair.get('b'),dict):
   anchor_observations.append({'pair':pair,'contact_observation':contact(c,pair['a'],pair['b'])})
  else:anchor_observations.append({'pair':pair,'contact_observation':result('unknown','anchor_pair_not_resolved')})
 return result('unknown','3d_topology_not_bound',{'order_view':{'status':order,'value':o,'scope':'world +Z visible surfaces'},'adjacent_contacts':{'status':gaps,'gaps_m':g,'pairs':rel.get('adjacent_pairs',[]),'scope':'existing finger-level search; not an inferred segment anchor'},'landing':{'status':landing,'margins_m':land},'declared_anchor_pair_proximity':anchor_observations,'contact_policy':rel.get('contact_policy','unknown'),'segment_level_topology':'unknown'})

def overlap(c,spec):
 """Declared frame directional ordering and one-dimensional projection proxy."""
 if not isinstance(spec,dict):return result('unknown','overlap_spec_not_resolved')
 a,b=spec.get('upper'),spec.get('lower');sa,sb=_segment(c,a),_segment(c,b)
 if sa is None or sb is None:return result('unknown','overlap_segment_not_resolved')
 frame=spec.get('frame');direction=spec.get('upper_direction');axis=spec.get('coverage_axis')
 if frame not in ('world','palm_local'):return result('unknown','overlap_frame_not_supported')
 try:
  direction=np.asarray(direction,dtype=float);axis=np.asarray(axis,dtype=float)
  if direction.shape!=(3,) or axis.shape!=(3,) or not np.isfinite(direction).all() or not np.isfinite(axis).all():raise ValueError()
  if min(np.linalg.norm(direction),np.linalg.norm(axis))<1e-12:raise ValueError()
  direction=direction/np.linalg.norm(direction);axis=axis/np.linalg.norm(axis)
 except (ValueError,TypeError):return result('unknown','overlap_axes_invalid')
 if abs(float(direction@axis))>1e-6:return result('unknown','overlap_axes_not_perpendicular')
 if frame=='palm_local':
  side={'left':0,'right':1}.get(spec.get('frame_hand'))
  if side is None:return result('unknown','overlap_frame_hand_missing')
  direction=c.hands[side]['rotation']@direction;axis=c.hands[side]['rotation']@axis
 bounds=spec.get('depth_range_m');minimum=spec.get('min_coverage_fraction')
 if not isinstance(bounds,list) or len(bounds)!=2 or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and np.isfinite(x) for x in bounds) or bounds[0]>bounds[1] or not isinstance(minimum,(int,float)) or isinstance(minimum,bool) or not np.isfinite(minimum) or not 0<=minimum<=1:return result('unknown','overlap_thresholds_not_specified')
 ia=sorted([float(sa[0]@axis),float(sa[1]@axis)]);ib=sorted([float(sb[0]@axis),float(sb[1]@axis)])
 if ib[1]-ib[0]<1e-12:return result('unknown','lower_projection_degenerate')
 coverage=max(0.,min(ia[1],ib[1])-max(ia[0],ib[0]))/(ib[1]-ib[0])
 pa,pb,_,_=segment_closest(sa[0],sa[1],sb[0],sb[1]);depth=float((pa-pb)@direction)
 local={'status':'pass' if bounds[0]<=depth<=bounds[1] and coverage>=minimum else 'fail','depth_m':depth,'depth_range_m':bounds,'lower_axis_coverage_fraction':coverage,'min_coverage_fraction':minimum}
 return result('unknown','local_overlap_proxy_only',{'proxy':local,'scope':'closest-axis depth and projected centerline interval only; not surface overlap, contact, or full regional ordering','requested':spec})

def sampled_overlap(c,spec):
 """Measure a declared capsule pair along the lower segment.

 A pass is only for the declared discrete capsule approximation and must not
 be promoted to continuous skin contact or physical reachability.
 """
 if not isinstance(spec,dict):return result('unknown','overlap_spec_not_resolved')
 try:
  if not _positions_match_q(c):return result('unknown','positions_not_consistent_with_fk')
 except (AttributeError,ValueError,TypeError,IndexError):return result('unknown','fk_consistency_not_checkable')
 a,b=spec.get('upper'),spec.get('lower');sa,sb=_segment(c,a),_segment(c,b)
 if sa is None or sb is None:return result('unknown','overlap_segment_not_resolved')
 direction=spec.get('upper_direction');frame=spec.get('frame');minimum=spec.get('min_coverage_fraction');num=spec.get('samples')
 if frame not in ('world','palm_local') or not isinstance(num,int) or isinstance(num,bool) or not 2<=num<=1000 or not isinstance(minimum,(int,float)) or isinstance(minimum,bool) or not np.isfinite(minimum) or not 0<minimum<=1:return result('unknown','overlap_parameters_invalid')
 try:
  direction=np.asarray(direction,float)
  if direction.shape!=(3,) or not np.isfinite(direction).all() or np.linalg.norm(direction)<1e-12:raise ValueError()
  direction/=np.linalg.norm(direction)
 except (TypeError,ValueError):return result('unknown','overlap_direction_invalid')
 if frame=='palm_local':
  side={'left':0,'right':1}.get(spec.get('frame_hand'))
  if side is None:return result('unknown','overlap_frame_hand_missing')
  direction=c.hands[side]['rotation']@direction
 tol=c.cfg.get('contact_tolerance_m');pen=c.cfg.get('penetration_tolerance_m')
 if not all(isinstance(x,(int,float)) and np.isfinite(x) and x>=0 for x in (tol,pen)):return result('unknown','contact_tolerance_missing')
 upper_a,upper_b,upper_radius=sa;lower_a,lower_b,lower_radius=sb;upper_vec=upper_b-upper_a;den=float(upper_vec@upper_vec)
 if den<1e-18:return result('unknown','upper_segment_degenerate')
 checks=[]
 for u in np.linspace(0.,1.,num):
  lower=lower_a+u*(lower_b-lower_a);t=float(np.clip((lower-upper_a)@upper_vec/den,0,1));upper=upper_a+t*upper_vec;delta=upper-lower;gap=float(np.linalg.norm(delta)-upper_radius-lower_radius);depth=float(delta@direction)
  checks.append({'u':float(u),'signed_gap_m':gap,'direction_m':depth,'sample_in_contact_and_direction':bool(-pen<=gap<=tol and depth>0)})
 coverage=float(np.mean([x['sample_in_contact_and_direction'] for x in checks]))
 return result('pass' if coverage>=minimum else 'fail','sampled_capsule_overlap_contract',{'sampled_coverage_fraction':coverage,'min_coverage_fraction':minimum,'samples':checks,'scope':'discrete declared capsule proximity and direction, not continuous skin-area proof','requested':spec})

def wrap(c,spec):
 """Sampled angular coverage around a declared target segment, not topology proof.

 Thresholds and sampling are supplied by the caller, never inferred from a seal.
 """
 if not isinstance(spec,dict):return result('unknown','wrap_spec_not_resolved')
 target=_segment(c,spec.get('target'));anchors=spec.get('wrapping_segments')
 required=spec.get('min_angular_coverage_rad');max_gap=spec.get('max_surface_gap_m');samples=spec.get('samples_per_segment')
 if target is None or not isinstance(anchors,list) or not anchors:return result('unknown','wrap_anchors_not_resolved')
 if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and np.isfinite(x) for x in (required,max_gap)) or not 0<=required<=2*np.pi or max_gap<0 or not isinstance(samples,int) or isinstance(samples,bool) or not 2<=samples<=100:
  return result('unknown','wrap_thresholds_or_sampling_not_specified')
 segments=[_segment(c,a) for a in anchors]
 if any(s is None for s in segments):return result('unknown','wrap_segment_not_resolved')
 a,b,r=target;axis=b-a;length=np.linalg.norm(axis)
 if length<1e-12:return result('unknown','wrap_axis_degenerate')
 axis=axis/length;basis=np.eye(3)[int(np.argmin(np.abs(axis)))];u=np.cross(axis,basis);u/=np.linalg.norm(u);v=np.cross(axis,u)
 angles=[];gaps=[]
 for p0,p1,radius in segments:
  for t in np.linspace(0.,1.,samples):
   p=p0+t*(p1-p0);axial=float((p-a)@axis)
   if not 0<=axial<=length:continue
   radial=p-a-axial*axis;distance=float(np.linalg.norm(radial));gap=distance-r-radius
   if distance>1e-12 and -c.cfg['penetration_tolerance_m']<=gap<=max_gap:
    angles.append(float(np.arctan2(radial@v,radial@u)%(2*np.pi)));gaps.append(gap)
 if not angles:coverage=0.
 else:
  angles.sort();coverage=float(2*np.pi-max(np.diff(angles+[angles[0]+2*np.pi])))
 evidence={'angular_coverage_rad':coverage,'min_angular_coverage_rad':required,'eligible_samples':len(angles),'surface_gaps_m':gaps,'sample_count_per_segment':samples,'scope':'sampled capsule axes near declared target segment; angular span is not continuous surface coverage or containment','requested':spec}
 if coverage<required:return result('unknown','sampled_proxy_below_threshold_wrap_not_proven',dict(evidence,proxy_status='fail'))
 return result('unknown','angular_proxy_passes_topology_not_bound',evidence)

def _joint_ranges(q, spec):
 """Only explicitly supplied joint ranges; no invented fist/relaxation threshold."""
 fingers=('index','middle','ring','pinky'); joints={'mcp':0,'pip':2,'dip':3}
 ranges=spec.get('joint_ranges_rad') if isinstance(spec,dict) else None
 if not isinstance(ranges,dict) or not ranges:return result('unknown','joint_ranges_not_specified')
 checks=[]
 for name,values in ranges.items():
  if name not in fingers or not isinstance(values,dict) or not values:return result('unknown','joint_range_not_supported')
  for joint,bounds in values.items():
   if joint not in joints or not isinstance(bounds,list) or len(bounds)!=2 or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and np.isfinite(v) for v in bounds) or bounds[0]>bounds[1]:return result('unknown','joint_range_not_supported')
   value=float(q[4*fingers.index(name)+joints[joint]])
   checks.append({'finger':name,'joint':joint,'value_rad':value,'range_rad':bounds,'pass':bool(bounds[0]<=value<=bounds[1])})
 return result('pass' if all(x['pass'] for x in checks) else 'fail','declared_joint_ranges',{'checks':checks,'scope':'declared ranges only'})

def _flex_gradient(q,spec):
 fingers=('index','middle','ring','pinky');joints={'mcp':0,'pip':2,'dip':3}
 if not isinstance(spec,dict) or spec.get('joint') not in joints or not isinstance(spec.get('pairs'),list) or not spec['pairs']:return result('unknown','gradient_definition_missing')
 checks=[]
 for pair in spec['pairs']:
  if not isinstance(pair,dict):return result('unknown','gradient_pair_not_supported')
  a,b=pair.get('from'),pair.get('to');bounds=pair.get('delta_range_rad')
  if a not in fingers or b not in fingers or not isinstance(bounds,list) or len(bounds)!=2 or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and np.isfinite(v) for v in bounds) or bounds[0]>bounds[1]:return result('unknown','gradient_pair_not_supported')
  j=joints[spec['joint']];v=float(q[4*fingers.index(b)+j]-q[4*fingers.index(a)+j])
  checks.append({'from':a,'to':b,'delta_rad':v,'range_rad':bounds,'pass':bool(bounds[0]<=v<=bounds[1])})
 return result('pass' if all(x['pass'] for x in checks) else 'fail','declared_flex_differences',{'joint':spec['joint'],'checks':checks})

def _positions_match_q(c):
 """Check that observed positions are the current q/wrist FK, without mutating c."""
 probe=copy.deepcopy(c)
 x=np.r_[probe._pack(probe.q[0]),probe._pack(probe.q[1]),probe.wrist_angles.ravel(),probe.wrist_delta]
 probe._apply_q(x)
 return bool(np.allclose(probe.p,c.p,rtol=0,atol=1e-9) and all(np.allclose(a,b,rtol=0,atol=1e-9) for a,b in zip(probe.q,c.q)))

def _relaxation(q,spec):
 """Explicit normalized distance from a specified joint-angle reference."""
 if not isinstance(spec,dict) or spec.get('method')!='weighted_joint_distance':return result('unknown','relaxation_measure_not_specified')
 terms=spec.get('terms');bounds=spec.get('range_rad')
 if not isinstance(terms,list) or not terms or not isinstance(bounds,list) or len(bounds)!=2 or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and np.isfinite(x) for x in bounds) or not 0<=bounds[0]<=bounds[1]:return result('unknown','relaxation_parameters_invalid')
 fingers=('index','middle','ring','pinky'); joints={'mcp':0,'pip':2,'dip':3};total=0.;weight=0.
 for term in terms:
  if not isinstance(term,dict) or term.get('finger') not in fingers or term.get('joint') not in joints:return result('unknown','relaxation_joint_unresolved')
  ref,w=term.get('reference_rad'),term.get('weight')
  if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and np.isfinite(x) for x in (ref,w)) or w<=0:return result('unknown','relaxation_reference_or_weight_invalid')
  val=float(q[4*fingers.index(term['finger'])+joints[term['joint']]])
  total+=w*(val-ref)**2;weight+=w
 value=float(np.sqrt(total/weight))
 return result('pass' if bounds[0]<=value<=bounds[1] else 'fail','declared_weighted_joint_distance',{'value_rad':value,'range_rad':bounds,'scope':'declared joint-distance measure, not physical muscle relaxation'})

def fist_opening(c,spec):
 """Expose the user-defined components; do not infer a hole from curl alone."""
 components=('fist_pose','flex_gradient','thumb_closure','opening_coordinates','relaxation','ring_rotation')
 if not isinstance(spec,dict):return result('unknown','fist_opening_spec_not_resolved',{'requested':spec})
 hand=spec.get('hand');side={'left':0,'right':1}.get(hand)
 evidence={'requested':spec,'components':{name:{'specified':name in spec and spec[name] is not None,'actual_status':'unknown'} for name in components},
           'geometry_and_thresholds':'not_bound','execution_and_contact':'not_verified'}
 if side is not None:
  q=c.q[side]
  positions_match_q=_positions_match_q(c)
  evidence['position_fk_consistent']=positions_match_q
  if positions_match_q:
   from fist_actual_metrics import measure
   measured=measure(c,hand);evidence['rig_fist_measurements']=measured
   if measured.get('status')=='measured':
    relaxation=spec.get('relaxation')
    if isinstance(relaxation,dict) and relaxation.get('method')=='mean_min_clearance_vs_standard':
     bounds=relaxation.get('range_m');value=measured['mean_min_clearance_difference_m']
     if isinstance(bounds,list) and len(bounds)==2 and all(isinstance(v,(float,int)) and not isinstance(v,bool) and np.isfinite(v) for v in bounds) and bounds[0]<=bounds[1]:
      obs=result('pass' if bounds[0]<=value<=bounds[1] else 'fail','declared_mean_clearance_difference',{'value_m':value,'range_m':bounds,'scope':measured['scope']})
      evidence['components']['relaxation']['observation']=obs;evidence['components']['relaxation']['actual_status']=obs['status']
    rotation=spec.get('ring_rotation')
    if isinstance(rotation,dict) and rotation.get('method')=='plane_angle_vs_standard':
     tolerance=rotation.get('tolerance_rad');angles=measured['plane_deviation_from_standard_rad']
     if isinstance(tolerance,(float,int)) and not isinstance(tolerance,bool) and np.isfinite(tolerance) and tolerance>=0 and all(v is not None for v in angles.values()):
      obs=result('pass' if all(abs(v)<=tolerance for v in angles.values()) else 'fail','declared_ring_plane_angles',{'plane_deviation_from_standard_rad':angles,'plane_to_wrist_angle_difference_rad':measured['plane_to_wrist_angle_difference_rad'],'tolerance_rad':tolerance,'scope':measured['scope']})
      evidence['components']['ring_rotation']['observation']=obs;evidence['components']['ring_rotation']['actual_status']=obs['status']
  evidence['actual_flex_angles_rad']={name:{'mcp':float(q[4*i]),'pip':float(q[4*i+2]),'dip':float(q[4*i+3])} for i,name in enumerate(('index','middle','ring','pinky'))}
  evidence['actual_flex_angles_note']='raw joint angles; neither flex gradient nor fist pose is validated'
  for name,observation in [('fist_pose',_joint_ranges(q,spec.get('fist_pose'))),('flex_gradient',_flex_gradient(q,spec.get('flex_gradient')))]:
   evidence['components'][name]['observation']=observation
   evidence['components'][name]['actual_status']=observation['status']
  if evidence['components']['relaxation']['actual_status']=='unknown':
   observation=_relaxation(q,spec.get('relaxation'))
   evidence['components']['relaxation']['observation']=observation
   evidence['components']['relaxation']['actual_status']=observation['status']
  standard=naruto.project_light(naruto.semantics_to_joints({**{name:'curled' for name in ('index','middle','ring','pinky')},'thumb':'curled','palm':'front'}))
  indices=[4*i+j for i in range(4) for j in (0,2,3)]
  evidence['standard_fist_difference']={'reference':'naruto curled template, MCP/PIP/DIP of four fingers','mean_signed_reference_minus_current_rad':float(np.mean(standard[indices]-q[indices])), 'mean_absolute_difference_rad':float(np.mean(np.abs(standard[indices]-q[indices]))), 'scope':'joint-space indicators only; not automatically a calibrated grip strength or minimum clearance'}
  closure=spec.get('thumb_closure')
  if isinstance(closure,dict):
   a,b=closure.get('a'),closure.get('b')
   if positions_match_q and isinstance(a,dict) and isinstance(b,dict) and a.get('hand')==b.get('hand')==hand and 'thumb' in (a.get('structure'),b.get('structure')) and a.get('structure')!=b.get('structure'):
    observed=contact(c,a,b)
    evidence['components']['thumb_closure']['observation']=observed
    evidence['components']['thumb_closure']['actual_status']=observed['status']
    evidence['components']['thumb_closure']['scope']='declared anchor contact only; not proof of a closed boundary'
  opening=spec.get('opening_coordinates')
  if positions_match_q and isinstance(opening,dict) and opening.get('method')=='rig_slice':
   from fist_rig_slice import inspect,inspect_segment_insertion
   obs=inspect(c,hand,opening.get('insertion_direction_world'),opening.get('resolution_m'),opening.get('margin_m'),opening.get('closure_tolerance_m'))
   evidence['components']['opening_coordinates']['observation']=obs
   if opening.get('accept_sampled_geometry') is True and opening.get('require_nonempty') is True and obs.get('status')=='sampled_candidate':
    evidence['components']['opening_coordinates']['actual_status']='pass' if all(x.get('projected_area_m2',0)>0 for x in obs['rings']) else 'fail'
   if 'insertion_probe' in opening:
    probe=opening['insertion_probe']
    if isinstance(probe,dict) and probe.get('point')=='minimum_ring_projection_center' and obs.get('status')=='sampled_candidate':
     insertion=inspect_segment_insertion(c,hand,probe.get('anchor'),obs['candidate_center_projection_world_m'],opening.get('insertion_direction_world'),probe.get('minimum_clearance_m'))
    else:insertion={'status':'unknown','reason':'insertion_probe_not_resolved'}
    evidence['components']['opening_coordinates']['insertion_probe']=insertion
    current=evidence['components']['opening_coordinates']['actual_status']
    if current!='fail':evidence['components']['opening_coordinates']['actual_status']=insertion['status'] if current=='pass' else 'unknown'
  if positions_match_q and isinstance(opening,dict) and opening.get('method')=='declared_ring_polygons':
   rings=opening.get('rings');axis=opening.get('insertion_direction_world');origin=opening.get('projection_origin_world_m');tol=opening.get('planarity_tolerance_m')
   wrist=c.hands[side]['wrist_b']-c.hands[side]['wrist_a']
   obs=measure_fist_rings(rings,axis,wrist,origin,tol)
   obs['rig_contour_binding']='unknown';obs['ring_points_source']='caller-provided, FK linkage not verified'
   evidence['components']['opening_coordinates']['observation']=obs
   if evidence['components']['ring_rotation']['actual_status']=='unknown':evidence['components']['ring_rotation']['observation']=obs
  if positions_match_q and isinstance(opening,dict) and opening.get('method')=='four_tip_centroid_proxy' and opening.get('frame') in ('world','palm_local'):
   tips=[_finger(c,{'hand':hand,'structure':name}) for name in ('index','middle','ring','pinky')]
   if all(tips):
    center=np.mean([c.p[f['ids'][-1]] for f in tips],axis=0)
    if opening['frame']=='palm_local':center=c.hands[side]['rotation'].T@(center-c.hands[side]['center'])
    evidence['components']['opening_coordinates']['observation']={'status':'unknown','proxy_point_m':center.tolist(),'frame':opening['frame'],'scope':'four fingertip centroid only; not a detected aperture, axis or size'}
 statuses=[v['actual_status'] for v in evidence['components'].values()]
 status='fail' if 'fail' in statuses else ('pass' if all(s=='pass' for s in statuses) else 'unknown')
 return result(status,'fist_opening_declared_components',dict(evidence,scope='declared components and specified geometry; not a continuous-path or anatomical-skin certificate'))

def contact_hold(before,after):
 """Endpoint-only guard; no continuous or sliding-path certificate."""
 if before['status']!='pass':return result('unknown','initial_contact_not_established',{'before':before,'after':after})
 if after['status']=='unknown':return result('unknown','final_contact_unknown',{'before':before,'after':after})
 return result('pass' if after['status']=='pass' else 'fail','endpoint_contact_hold_only',{'before':before,'after':after,'continuous_path':'unknown'})

def evaluate_contracts(c,contracts,baseline=None):
 """Evaluate explicit relations without turning unknown into achieved facts."""
 if not isinstance(contracts,list) or not contracts:return result('unknown','relation_contracts_missing')
 observations=[]
 for contract in contracts:
  if not isinstance(contract,dict) or not isinstance(contract.get('spec'),dict):
   observations.append(result('unknown','invalid_relation_contract'));continue
  kind=contract.get('kind')
  from relation_observer import observe
  obs=observe(c,contract,baseline)
  observations.append({'kind':kind,'observation':obs})
 statuses=[x.get('observation',x)['status'] for x in observations]
 return result('fail' if 'fail' in statuses else ('unknown' if 'unknown' in statuses else 'pass'),'declared_relation_contracts',{'observations':observations,'scope':'endpoint checks only; statuses reflect each check scope; continuous_path unknown'})

def evaluate(c):
 """Observe only the relations actually declared in current gesture."""
 opening=c.seal.get('fist_opening')
 wrap_spec=c.seal.get('wrap')
 overlap_spec=c.seal.get('overlap')
 overlap_result=sampled_overlap(c,overlap_spec) if isinstance(overlap_spec,dict) and overlap_spec.get('method')=='sampled_capsules' else overlap(c,overlap_spec) if overlap_spec is not None else result('unknown','no_executable_overlap_specification')
 relations={'interlock':interlock(c),'contacts':[], 'overlap':overlap_result, 'wrap':wrap(c,wrap_spec) if wrap_spec is not None else result('unknown','no_executable_wrap_specification'), 'fist_opening':fist_opening(c,opening) if opening is not None else result('unknown','no_executable_opening_specification')}
 gaps=c._contact_gaps().tolist();tol=c.cfg['contact_tolerance_m'];pen=c.cfg['penetration_tolerance_m']
 for spec,gap in zip(c.seal['contacts'],gaps):
  relations['contacts'].append({'spec':spec,**result('pass' if -pen<=gap<=tol else 'fail','contact_band',{'signed_gap_m':gap,'contact_tolerance_m':tol,'penetration_tolerance_m':pen,'scope':'existing gesture_feedback contact'})})
 failed=[x for x in relations['contacts'] if x['status']=='fail']
 failed_opening=relations['fist_opening']['status']=='fail'; failed_wrap=relations['wrap']['status']=='fail';failed_overlap=relations['overlap']['status']=='fail'
 report={'schema':'naruto.semantic_relation_observation.v1','seal':c.seal.get('id'),'relations':relations,
         'overall':'fail' if failed or failed_opening or failed_wrap or failed_overlap else 'unknown',
         'reason':'declared relation check failed' if failed or failed_opening or failed_wrap or failed_overlap else 'complete semantic and 3D topology not yet observed',
         'exception':{'facts':{'failed_contacts':failed,'fist_opening':relations['fist_opening'],'interlock':relations['interlock'],'continuous_path':'unknown'},'candidate_levers':[]},
         'continuous_path':'unknown'}
 from relation_observer import observe_many
 contracts=[{'kind':'contact','spec':s} for s in c.seal['contacts']]
 for kind in ('interlock','wrap','fist_opening','overlap'):
  if c.seal.get(kind):contracts.append({'kind':kind,'spec':c.seal[kind]})
 normalized=observe_many(c,contracts)
 report['normalized_checks']=normalized;report['overall']=normalized['status']
 report['target_verification']=copy.deepcopy(c.seal.get('target_verification'))
 if (c.seal.get('target_verification') or {}).get('status')=='unknown' and report['overall']=='pass':report['overall']='unknown'
 report['reason']='declared current-state relation checks; preservation and continuous path evaluated separately'
 report['exception']['normalized_exceptions']=normalized['exceptions']
 return report
