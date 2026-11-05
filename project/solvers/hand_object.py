"""Budgeted local least squares on the existing spring FK and anatomy controls.

No pose enumeration, random seeds, root edits, new hand axes, or physics claims.
"""
import copy,json,math,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares,minimize_scalar
import anatomy_authority as anatomy
from gesture_feedback import Feedback
from gesture_pipeline import build_semantic_seal
from animation_model.rigs.spring import Assembler
from animation_model.geometry import segment_closest,box_sdf,segment_box
from assembly.contracts import RunBudget
from compiler.hand_object import compile_hand_object,normal_cylinder_reach_guard,DIGITS
from assembly.io import fingerprint

class StopBudget(Exception):pass

def sdf_cylinder(points,axisxy,radius,span):
    p=np.asarray(points);rad=np.linalg.norm(p[...,:2]-axisxy,axis=-1)-radius
    # Axis is palmar -Z; span contains positive distances along that axis.
    z=-p[...,2];mid=sum(span)/2;half=(span[1]-span[0])/2
    d=np.stack([rad,np.abs(z-mid)-half],axis=-1)
    return np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(np.max(d,axis=-1),0)

def segment_cylinder(a,b,axisxy,radius,span):
    # Minimize signed distance along a segment, using the convex axial pieces.
    # This is collision sensing, not additional FK or joint-space search.
    edges=[0.,1.];dz=-(b[2]-a[2])
    if abs(dz)>1e-15:
      for z in (*span,sum(span)/2):
        u=(z+a[2])/dz
        if 0<u<1:edges.append(float(u))
    delta=b-a;den=delta[:2]@delta[:2]
    if den>1e-18:
      u=float(np.clip(-((a[:2]-axisxy)@delta[:2])/den,0,1));edges.append(u)
    edges=sorted(set(edges));candidates=[(float(sdf_cylinder(a+u*delta,axisxy,radius,span)),u) for u in edges]
    for lo,hi in zip(edges,edges[1:]):
      if hi-lo<1e-8:continue
      r=minimize_scalar(lambda u:float(sdf_cylinder(a+u*delta,axisxy,radius,span)),bounds=(lo,hi),method='bounded',options={'xatol':1e-7,'maxiter':30})
      candidates.append((float(r.fun),float(r.x)))
    return min(candidates)

def lateral_segment_gap(a,b,axisxy,radius,finger_radius,span):
    """Side contact only. End-cap proximity cannot satisfy this predicate."""
    v=b-a;z=-a[2];dz=-v[2]
    if abs(dz)<1e-15:
      if not span[0]<=z<=span[1]:
        return float(np.linalg.norm(a[:2]-axisxy)-radius-finger_radius+max(span[0]-z,z-span[1],0.)),a.copy(),False
      lo,hi=0.,1.
    else:
      limits=sorted([(span[0]-z)/dz,(span[1]-z)/dz]);lo=max(0.,limits[0]);hi=min(1.,limits[1])
      if lo>hi:
        # Explicit axial miss drives approach but is never a certified contact.
        u=float(np.clip((sum(span)/2-z)/dz,0,1));p=a+u*v
        return float(abs(np.linalg.norm(p[:2]-axisxy)-radius-finger_radius)+max(span[0]+p[2],-p[2]-span[1],0.)),p,False
    den=v[:2]@v[:2];u=float(np.clip(-((a[:2]-axisxy)@v[:2])/den,lo,hi)) if den>1e-18 else lo
    p=a+u*v;return float(np.linalg.norm(p[:2]-axisxy)-radius-finger_radius),p,True

class HandObjectRun:
    def __init__(self,request):
      self.request=request;self.seal=build_semantic_seal('snake');self.model=self.seal['kinematics']
      self.h=self.model['hands'][0];self.backend=Assembler.__new__(Assembler);self.backend.model=self.model
      self.program=compile_hand_object(request,self.model,anatomy.AUTHORITY)
      self.budget=RunBudget(fk=request['budget'],abstract=3);self.budget.charge('abstract',3)
      self.start=time.monotonic();self.cache={};self.trace=[];self.current_mode='seed';self.phase_start=0;self.quota=1
      self.R=request['object']['circumference_m']/(2*np.pi);self.span=request['object']['axial_span_m']
      self.tol=request['contact_tolerance_m'];self.pen=request['penetration_tolerance_m']
      q=anatomy.grip_controls(.45)
      # The physiological cascade and explicit inward Spread initialize the one branch.
      q[[1,5,9,13]]=[.30,.12,-.12,-.30]
      q[16:]=[np.deg2rad(55),np.deg2rad(30),0.,.5,.5];q=anatomy.project(q)
      self.seedq=q.copy();self.seed=np.r_[Feedback._pack(q),.020,0.]
      independent=[4*i+k for i in range(4) for k in range(3)]+[16,17,19,20]
      bounds=anatomy.LIMITS_RAD[independent]
      self.lo=np.r_[bounds[:,0],-.025,-.055];self.hi=np.r_[bounds[:,1],.125,.055]
      # 16 intrinsic controls + 2 relation offsets; scaled, not dimensionally mixed.
      self.scale=np.r_[np.ones(16),.05,.05]
      self.best={};self.rows=[]

    def evaluate(self,x):
      key=tuple(np.round(x,12))
      if key in self.cache:return self.cache[key]
      if self.budget.used['FK']-self.phase_start>=self.quota or self.budget.used['FK']>=self.budget.limits['FK']:raise StopBudget('FK budget')
      if time.monotonic()-self.start>=120:raise StopBudget('wall clock')
      self.budget.charge('FK',1)
      q=Feedback._unpack(np.asarray(x[:16]));q=anatomy.project(q)
      local=self.backend.local(0,q);axisxy=np.asarray(x[16:18]);segments=[]
      for name in DIGITS:
        f=self.h['fingers'][name];p=local[name]
        for j in range(f.get('surface_segment_start',0),3):segments.append((name,j,p[j],p[j+1],f['radius_m']))
      object_gaps=[];object_points=[]
      for name,j,a,b,r in segments:
        d,u=segment_cylinder(a,b,axisxy,self.R,self.span);object_gaps.append(d-r);object_points.append(a+(b-a)*u)
      self_gaps=[]
      for i,(name,j,a,b,r) in enumerate(segments):
        for other,k,c,d,s in segments[i+1:]:
          if name==other and abs(j-k)<=1:continue
          pa,pb,_,_=segment_closest(a,b,c,d)
          if name!=other and all(box_sdf(p,np.array(self.h['palm']['size_m'])/2)[0]<0 for p in (pa,pb)):
            self_gaps.append(1.);continue
          self_gaps.append(np.linalg.norm(pa-pb)-r-s)
      palm_gaps=[];half=np.array(self.h['palm']['size_m'])/2
      for name,j,a,b,r in segments:
        aa=a.copy();embedded=self.h['fingers'][name].get('embedded_base',False)
        if j==0 or (embedded and j==1):
          da=box_sdf(a,half)[0];db=box_sdf(b,half)[0]
          if da<r:
            if db<=r:palm_gaps.append(1.);continue
            lo,hi=0.,1.
            for _ in range(18):
              mid=(lo+hi)/2
              if box_sdf(a+(b-a)*mid,half)[0]<r:lo=mid
              else:hi=mid
            aa=a+(b-a)*hi
        d,_,_=segment_box(aa,b,np.zeros(3),np.eye(3),half);palm_gaps.append(d-r)
      # Palm/wrist are wholly before the near cap (a hard geometric exclusion).
      # Still inspect every visible finger against the existing wrist capsule.
      wrist=np.array([[-.08,0.,0.],[-.04,0.,0.]])
      for name,j,a,b,r in segments:
        pa,pb,_,_=segment_closest(a,b,*wrist);palm_gaps.append(np.linalg.norm(pa-pb)-r-self.h['wrist']['radius_m'])
      tip_gaps=[]
      for name in DIGITS:
        p=local[name][-1];r=self.h['fingers'][name]['radius_m'];rad=np.linalg.norm(p[:2]-axisxy)-self.R-r
        # Side contact is restricted to the actual finite shaft interval.
        axial=max(self.span[0]+r-(-p[2]),(-p[2])-(self.span[1]-r),0.)
        tip_gaps.append(float(np.hypot(max(rad,0.),axial) if rad>=0 else rad-axial))
      chain_gaps={};chain_points={};side_eligible={}
      for name in DIGITS:
        ids=[i for i,s in enumerate(segments) if s[0]==name]
        candidates=[lateral_segment_gap(segments[i][2],segments[i][3],axisxy,self.R,segments[i][4],self.span) for i in ids]
        eligible=[c for c in candidates if c[2]]
        g,p,e=min(eligible or candidates,key=lambda c:abs(c[0]));chain_gaps[name]=g;chain_points[name]=p;side_eligible[name]=e
      tips=np.array([local[n][-1] for n in DIGITS]);perimeter=float(np.linalg.norm(tips-np.roll(tips,-1,axis=0),axis=1).sum())
      row={'q':q,'local':local,'x':np.asarray(x).copy(),'segments':segments,'object_gaps':np.array(object_gaps),
        'object_points':np.array(object_points),'self_gaps':np.array(self_gaps),'palm_gaps':np.array(palm_gaps),
        'tip_gaps':np.array(tip_gaps),'chain_gaps':chain_gaps,'chain_points':chain_points,'side_eligible':side_eligible,'tip_perimeter_m':perimeter,
        'evaluation':self.budget.used['FK'],'mode_evaluated':self.current_mode}
      self.cache[key]=row
      if hasattr(self,'out'):
        with (self.out/'FK_ledger.jsonl').open('a') as ledger:
          ledger.write(json.dumps({'evaluation':row['evaluation'],'mode':self.current_mode,'x':row['x'].tolist(),'q':q.tolist()})+'\n')
      return row

    def residual(self,z,mode):
      row=self.evaluate(z*self.scale);q=row['q'];x=row['x'];contract=self.program['contracts'][mode]
      out=[]
      if mode=='pinch':out.extend(row['tip_gaps']/.001)
      else:out.extend(np.array([row['chain_gaps'][n] for n in contract['contact_digits']])/.001)
      # All collision pairs remain active even when they do not define a contact.
      for gaps in [row['object_gaps'],row['self_gaps'],row['palm_gaps']]:out.extend(5*np.maximum(-gaps-self.pen,0)/.001)
      if mode in ('wrap','grasp'):
        names=contract['contact_digits'];v=np.array([row['chain_points'][n][:2]-x[16:18] for n in names]);v/=np.maximum(np.linalg.norm(v,axis=1)[:,None],1e-12)
        # Opposing normals are a smooth target; the existing observer judges span.
        if mode=='wrap':out.extend(2*(v[0]+v[1]))
        else:out.extend(2*v.mean(axis=0))
      if mode=='pinch':out.append(.18*row['tip_perimeter_m']/.02)
      # Existing q reference is only a tie-breaker, never anatomical evidence.
      out.extend(.025*(q-self.seedq));out.extend(.02*(x[16:18]-self.seed[16:18])/.05)
      residual=np.asarray(out);score=float(residual@residual)
      if mode not in self.best or score<self.best[mode][0]:self.best[mode]=(score,row)
      self.trace.append({'evaluation':row['evaluation'],'mode':mode,'objective':score,'cached':row['mode_evaluated']!=mode,
        'max_object_penetration_mm':float(max(0.,-row['object_gaps'].min())*1000),
        'max_self_penetration_mm':float(max(0.,-row['self_gaps'].min())*1000)})
      return residual

    def observe(self,row,mode):
      from semantic_relations import wrap
      # Bind the cylinder axis as an explicit external segment to the existing observer.
      from types import SimpleNamespace
      c=SimpleNamespace(p=[],fingers=[],model={'hands':[]},cfg={'contact_tolerance_m':self.tol,'penetration_tolerance_m':self.pen})
      wrapping=[]
      for name in DIGITS:
        ids=list(range(len(c.p),len(c.p)+4));c.p.extend(row['local'][name]);c.fingers.append({'side':0,'name':name,'ids':ids,'radius':self.h['fingers'][name]['radius_m']})
        # Embedded-base material can be included in collision sensing while the
        # existing wrap anchor protocol deliberately excludes that internal link.
        first=self.h['fingers'][name].get('surface_segment_start',0)
        if name=='thumb' and self.h['fingers'][name].get('embedded_base',False):first=max(first,1)
        for j in range(first,3):wrapping.append({'hand':'left','structure':name,'region':{'type':'segment','id':f'segment_{j+1}'}})
      xy=row['x'][16:18];p0=np.r_[xy,-self.span[0]];p1=np.r_[xy,-self.span[1]]
      ids=list(range(len(c.p),len(c.p)+4));c.p.extend([p0,p1,p1,p1]);c.fingers.append({'side':1,'name':'index','ids':ids,'radius':self.R});c.p=np.array(c.p)
      c.model['hands']=[self.h,{'fingers':{'index':{}}}]
      spec={'target':{'hand':'right','structure':'index','region':{'type':'segment','id':'segment_1'}},'wrapping_segments':wrapping,
        'min_angular_coverage_rad':np.pi,'max_surface_gap_m':self.tol,'samples_per_segment':25}
      observation=wrap(c,spec)
      contract=self.program['contracts'][mode];gaps=row['tip_gaps'] if mode=='pinch' else np.array([row['chain_gaps'][n] for n in contract['contact_digits']])
      contact=(gaps>=-self.pen)&(gaps<=self.tol)
      if mode!='pinch':contact &= np.array([row['side_eligible'][n] for n in contract['contact_digits']])
      penetration=max(0.,-row['object_gaps'].min(),-row['self_gaps'].min(),-row['palm_gaps'].min())
      angle=observation['evidence']['angular_coverage_rad']
      proxy=bool(contact.all() and penetration<=self.pen+1e-9 and (mode=='pinch' or angle>=np.pi-1e-9))
      return {'relation':mode,'status':'sampled_geometric_proxy_pass' if proxy else 'not_achieved',
        'contact_digits':contract['contact_digits'],'contact_count':int(contact.sum()),'required_contact_count':len(contact),
        'contact_gaps_mm':dict(zip(contract['contact_digits'],(gaps*1000).tolist())),
        'all_digit_chain_gaps_mm':{k:v*1000 for k,v in row['chain_gaps'].items()},
        'all_tip_gaps_mm':dict(zip(DIGITS,(row['tip_gaps']*1000).tolist())),
        'max_penetration_mm':float(penetration*1000),'object_penetration_mm':float(max(0.,-row['object_gaps'].min())*1000),
        'self_penetration_mm':float(max(0.,-row['self_gaps'].min())*1000),'palm_wrist_penetration_mm':float(max(0.,-row['palm_gaps'].min())*1000),
        'theta_deg':0.,'cylinder_axis_point_in_palm_m':p0.tolist(),'tip_center_cycle_perimeter_mm':row['tip_perimeter_m']*1000,
        'sampled_angular_span_deg':float(np.degrees(angle)),'equivalent_angular_span_arc_mm':float(angle*self.R*1000),
        'angular_span_is_continuous_contact_coverage':False,'wrap_observation':observation,
        'anatomy':anatomy.check(row['q']),'physical_acceptance':False,'force_closure':'not_evaluated',
        'global_minimum_B':'not_established','hand_self_collision_scope':'current rigid capsules and palm box; no skin',
        'topological_enclosure':'unknown','continuous_approach_path':'not_computed'}

    def export(self,row,mode,observation):
      q=row['q'];hands=[]
      for side in ['left']:
        shift=np.zeros(3) if side=='left' else np.array([.35,0.,0.])
        fingers={n:{'points_world_m':(row['local'][n]+shift).tolist(),'radius_m':self.h['fingers'][n]['radius_m'],
          'surface_segment_start':self.h['fingers'][n].get('surface_segment_start',0)} for n in DIGITS}
        hands.append({'side':side,'kinematic_authority':'legacy_spring_feedback','control_q_rad':q.tolist(),
          'palm_center_world_m':shift.tolist(),'palm_rotation_matrix':np.eye(3).tolist(),'palm_size_m':self.h['palm']['size_m'],
          'wrist_world_m':(np.array([[-.08,0.,0.],[-.04,0.,0.]])+shift).tolist(),'fingers':fingers,
          'role':'solved' if side=='left' else 'inactive_schema_placeholder_not_rendered','control_validation':anatomy.check(q)})
      xy=row['x'][16:18]
      return {'schema':'gesture.feedback_result.v1','authority':'gesture.py','kinematic_authority':'legacy_spring_feedback',
        'status':observation['status'],'hand_scope':'single_hand_object','hands':hands,'q_left':q.tolist(),
        'metrics':{'accepted':False,'physical_assessment':'spring_hand_authoritative','penetration_m':observation['max_penetration_mm']/1000,
          'contact_error_m':max(abs(v) for v in observation['contact_gaps_mm'].values())/1000,'physical_stability':'not_evaluated'},
        'target_contract':{'id':'d25_theta0_'+mode,'pose_variant':mode,'current_locked_target_description':mode+'; diameter=25mm; theta=0',
          'declared_relations':self.program['contracts'][mode],'measured':observation},
        'brush_shaft':{'radius_m':self.R,'axis_point_world_m':[*xy,0.],'axis_direction_world_m':[0.,0.,-1.],'span_m':self.span},
        'render_hints':{'active_hand':'left','hand_color':'orange','hide_inactive_placeholder':True,'show_palm_box_outline':True},
        'provenance':{'FK':'animation_model.rigs.spring.Assembler.local','geometry':'gesture_pipeline.build_semantic_seal(snake), unchanged roots/lengths',
          'model_sha256':fingerprint(self.model),'anatomy_sha256':anatomy.SHA256,'evaluation':row['evaluation'],
          'inactive_hand':'placeholder only; it is not a mirrored anatomical result'},
        'positions_m':[p for n in DIGITS for p in row['local'][n].tolist()]}

    def run(self,out):
      out=Path(out);out.mkdir(parents=True,exist_ok=True);self.out=out
      if (out/'FK_ledger.jsonl').exists():raise RuntimeError('Existing ledger: budget must not be reset')
      def dump(name,data):(out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
      dump('semantic_program.json',self.program);dump('model_snapshot.json',self.model)
      guard=normal_cylinder_reach_guard(self.model,self.request['object']['circumference_m'],self.tol);dump('analytic_reach_guard.json',guard)
      initial=self.evaluate(self.seed);dump('initial_seed_q.json',{'q_rad':initial['q'].tolist(),'anatomy':anatomy.check(initial['q'])})
      warm=self.seed.copy();summary=[]
      for mode,quota in self.program['schedule']['per_candidate_quotas'].items():
        self.current_mode=mode;self.phase_start=self.budget.used['FK'];self.quota=quota
        # One continuation branch per contract; deterministic, no restarts.
        start=warm.copy()
        if mode=='wrap':start[16:18]=[.020,0.]
        lo=self.lo.copy();hi=self.hi.copy()
        if mode in ('grasp','pinch'):
          # Bind the analytic five-contact reach guard BEFORE local optimization.
          branches=guard['axis_x_required_branches_m']
          if guard['thumb_reach_requires_axis_x_at_most_m']<branches['positive_min']:
            hi[16]=min(hi[16],branches['negative_max'])
          lo[16]=max(lo[16],guard['index_reach_requires_axis_x_at_least_m'])
        start=np.clip(start,lo+1e-10,hi-1e-10)
        status=''
        try:
          opt=least_squares(lambda z:self.residual(z,mode),start/self.scale,bounds=(lo/self.scale,hi/self.scale),
            jac='2-point',diff_step=2e-5,max_nfev=quota,ftol=1e-6,xtol=1e-6,gtol=1e-5,x_scale='jac')
          status=opt.message
        except StopBudget as exc:status=str(exc)
        if mode not in self.best:
          self.residual(self.seed/self.scale,mode)
        score,row=self.best[mode];warm=row['x'].copy();obs=self.observe(row,mode)
        obs['termination']=status;obs['FK_evaluations_in_phase']=self.budget.used['FK']-self.phase_start
        obs['analytic_five_contact_guard']=guard if mode in ('grasp','pinch') else None
        dump(mode+'_result.json',self.export(row,mode,obs));dump(mode+'_assessment.json',obs)
        summary.append(obs)
        print(json.dumps({'mode':mode,'status':obs['status'],'contacts':str(obs['contact_count'])+'/'+str(obs['required_contact_count']),
          'penetration_mm':obs['max_penetration_mm'],'span_deg':obs['sampled_angular_span_deg'],'FK_used':self.budget.used['FK']},ensure_ascii=False),flush=True)
      report={'schema':'naruto.hand_object_experiment.v1','summary':summary,'analytic_guard':guard,'budget':self.budget.report(),
        'trace_entries':len(self.trace),'unique_cached_states':len(self.cache),'elapsed_seconds':time.monotonic()-self.start,
        'random_search_calls':0,'restarts':0,'structural_candidates':3,'geometry_edited':False,'physical_acceptance':False,
        'important':'Failure is not repaired by changing theta, cylinder circumference, roots, bone lengths, ROM, or contact requirements.'}
      dump('report.json',report);dump('optimization_trace.json',self.trace)
      return report
