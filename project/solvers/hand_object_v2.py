"""Structured continuation with analytic FK derivatives and constrained sensing.

FK remains Assembler.local. Jacobians differentiate its existing rigid chains;
they do not replace the model or spend hidden finite-difference pose probes.
"""
import copy,json,math,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import naruto,anatomy_authority as anatomy
from gesture_feedback import Feedback
from animation_model.geometry import segment_closest,box_sdf,segment_box
from compiler.hand_object import DIGITS,normal_cylinder_reach_guard
from solvers.hand_object import HandObjectRun,StopBudget,segment_cylinder,lateral_segment_gap

def point_jacobians(model,row):
    """d(saved FK points)/d(16 independent angles + two object offsets)."""
    h=model['hands'][0];q=row['q'];out={}
    for name in DIGITS:
      spec=h['fingers'][name];M=np.array(spec.get('finger_chirality_matrix',h.get('finger_chirality_matrix',np.eye(3))))
      p=row['local'][name];canonical=(p-p[0])@M
      J=np.zeros((4,3,18))
      if name!='thumb':
        i=['index','middle','ring','pinky'].index(name);abd=q[4*i+1]
        axis=naruto.rz(abd)[:3,:3]@np.array([0.,1.,0.])
        for k in range(1,4):
          J[k,:,3*i]=M@np.cross(axis,canonical[k])
          J[k,:,3*i+1]=M@np.cross([0.,0.,1.],canonical[k])
          if k>=2:J[k,:,3*i+2]=M@np.cross(axis,canonical[k]-canonical[1])
          if k>=3:J[k,:,3*i+2]+=anatomy.RATIO*(M@np.cross(axis,canonical[k]-canonical[2]))
      else:
        fe,aa,ps,mcp,ip=q[16:21];cmc=naruto.thumb_cmc_transform(fe,aa,ps)[:3,:3]
        yaw=np.deg2rad(-naruto.THUMB_REST_PRIOR['radial_spread_deg'])+fe
        aa_axis=naruto.rz(yaw)[:3,:3]@np.array([0.,1.,0.]);roll=cmc[:,0];hinge=cmc[:,2]
        for k in range(1,4):
          J[k,:,12]=M@(np.cross([0.,0.,1.],canonical[k])-anatomy.D_PS*np.cross(roll,canonical[k]))
          J[k,:,13]=M@(-np.cross(aa_axis,canonical[k])-anatomy.E_PS*np.cross(roll,canonical[k]))
          if k>=2:J[k,:,14]=M@np.cross(hinge,canonical[k]-canonical[1])
          if k>=3:J[k,:,15]=M@np.cross(hinge,canonical[k]-canonical[2])
      out[name]=J
    return out

def cylinder_normal(p,xy,R,span):
    v=p[:2]-xy;rho=np.linalg.norm(v);nr=np.r_[v/max(rho,1e-15),0.]
    dr=rho-R;s=-p[2];mid=sum(span)/2;dz=abs(s-mid)-(span[1]-span[0])/2
    nz=np.array([0.,0.,-1. if s>=mid else 1.])
    if dr>0 and dz>0:return (dr*nr+dz*nz)/np.hypot(dr,dz)
    return nr if dr>=dz else nz

class StructuredHandObjectRun(HandObjectRun):
    def __init__(self,request):
      super().__init__(request)
      self.program['schedule'].update({'seed_evaluations':19,'per_candidate_quotas':{'wrap':140,'grasp':190,'pinch':150},
        'counting':'19 explicitly charged derivative-audit FK states + constrained optimizer states; analytic Jacobian/observers reuse each FK state',
        'stopping':['contract reached','SLSQP local convergence','phase quota','500 shared FK','120 seconds wall clock']})
      self.program['contracts']['wrap']['contact_digits']=['index','pinky']
      self.program['contracts']['wrap']['anchor_choice']='outer digits give the half-surround span; intermediate digits remain collision checked'
      self.program['topology_branch']={'outer_contact_order':['index','middle','ring','pinky'],
        'outer_target_angles_deg':{'index':100.,'pinky':-100.},
        'axial_staggering':'free through existing finger joint controls; no root shifts',
        'guard_binding':'necessary reach branch is an actual optimizer bound for five-contact modes',
        'branch_count':1,'enumerations':0}
      self.program['morphemes']['TouchObject']['side_endcap_separation']='side-only observer and objective; caps used exclusively for collision'
      self.program['derivative_method']='analytic serial-chain Jacobians checked against 18 charged one-sided FK perturbations'
      self.deriv_cache={};self.phase_keys={};self.current_mode='derivative_audit';self.quota=19
      self.runlabel='d25_theta0_corrected_r2'

    def geometry(self,row):
      key=row['evaluation']
      if key in self.deriv_cache:return self.deriv_cache[key]
      Js=point_jacobians(self.model,row);xy=row['x'][16:18]
      O=np.zeros((2,18));O[0,16]=1;O[1,17]=1
      segs=row['segments'];object_grad=[];self_grad=[];palm_grad=[];half=np.array(self.h['palm']['size_m'])/2
      for name,j,a,b,r in segs:
        d,u=segment_cylinder(a,b,xy,self.R,self.span);p=a+u*(b-a)
        Jp=(1-u)*Js[name][j]+u*Js[name][j+1];n=cylinder_normal(p,xy,self.R,self.span)
        object_grad.append(n@Jp-n[:2]@O)
      for i,(name,j,a,b,r) in enumerate(segs):
        for other,k,c,d,s in segs[i+1:]:
          if name==other and abs(j-k)<=1:continue
          pa,pb,u,v=segment_closest(a,b,c,d)
          if name!=other and all(box_sdf(p,half)[0]<0 for p in (pa,pb)):self_grad.append(np.zeros(18));continue
          delta=pa-pb;n=delta/max(np.linalg.norm(delta),1e-15)
          self_grad.append(n@((1-u)*Js[name][j]+u*Js[name][j+1]-(1-v)*Js[other][k]-v*Js[other][k+1]))
      for name,j,a,b,r in segs:
        aa=a.copy();s0=0.;embedded=self.h['fingers'][name].get('embedded_base',False)
        if j==0 or (embedded and j==1):
          da=box_sdf(a,half)[0];db=box_sdf(b,half)[0]
          if da<r:
            if db<=r:palm_grad.append(np.zeros(18));continue
            lo,hi=0.,1.
            for _ in range(18):
              mid=(lo+hi)/2
              if box_sdf(a+(b-a)*mid,half)[0]<r:lo=mid
              else:hi=mid
            s0=hi;aa=a+(b-a)*hi
        _,t,n=segment_box(aa,b,np.zeros(3),np.eye(3),half);t=s0+(1-s0)*t
        palm_grad.append(n@((1-t)*Js[name][j]+t*Js[name][j+1]))
      wrist=np.array([[-.08,0.,0.],[-.04,0.,0.]])
      for name,j,a,b,r in segs:
        pa,pb,u,_=segment_closest(a,b,*wrist);n=(pa-pb)/max(np.linalg.norm(pa-pb),1e-15)
        palm_grad.append(n@((1-u)*Js[name][j]+u*Js[name][j+1]))
      chain={};chainpoint={};chaingrad={};chainpointgrad={};eligible={}
      for name in DIGITS:
        specs=[]
        for n,j,a,b,r in segs:
          if n!=name:continue
          gap,p,e=lateral_segment_gap(a,b,xy,self.R,r,self.span)
          v=b-a;den=v@v;u=float(np.clip(((p-a)@v)/den,0,1)) if den else 0.
          Jp=(1-u)*Js[name][j]+u*Js[name][j+1]
          if e and 1e-7<u<1-1e-7 and abs(v[2])>1e-12 and min(abs(-p[2]-z) for z in self.span)<1e-7:
            Jp=Jp-np.outer(v,Jp[2]/v[2])
          radial=p[:2]-xy;rho=np.linalg.norm(radial);g=(radial/max(rho,1e-15))@(Jp[:2]-O)
          if not e:
            if rho-self.R-r<0:g=-g
            z=-p[2]
            if z<self.span[0]:g+=Jp[2]
            elif z>self.span[1]:g-=Jp[2]
          specs.append((gap,p,e,g,Jp))
        good=[s for s in specs if s[2]];s=min(good or specs,key=lambda s:abs(s[0]))
        chain[name],chainpoint[name],eligible[name],chaingrad[name],chainpointgrad[name]=s
      tip=[];tipgrad=[];axial=[];axialgrad=[]
      for name in DIGITS:
        p=row['local'][name][-1];Jp=Js[name][-1];v=p[:2]-xy;rho=np.linalg.norm(v)
        tip.append(rho-self.R-self.h['fingers'][name]['radius_m']);tipgrad.append((v/max(rho,1e-15))@(Jp[:2]-O))
        axial.extend([-p[2]-self.span[0],self.span[1]+p[2]]);axialgrad.extend([-Jp[2],Jp[2]])
      result={'J':Js,'chain':chain,'chainpoint':chainpoint,'chaingrad':chaingrad,'chainpointgrad':chainpointgrad,'eligible':eligible,
        'tip':np.array(tip),'tipgrad':np.array(tipgrad),'axial':np.array(axial),'axialgrad':np.array(axialgrad),
        'collision':np.r_[row['object_gaps'],row['self_gaps'],row['palm_gaps']],
        'collisiongrad':np.vstack([object_grad,self_grad,palm_grad]),'O':O}
      self.deriv_cache[key]=result;return result

    def get(self,z,mode):
      row=self.evaluate(np.asarray(z)*self.scale)
      self.phase_keys.setdefault(mode,set()).add(tuple(np.round(row['x'],12)))
      return row,self.geometry(row)

    def objective_data(self,row,g,mode):
      names=self.program['contracts'][mode]['contact_digits']
      residual=[];jac=[]
      gaps=g['tip'] if mode=='pinch' else np.array([g['chain'][n] for n in names])
      dgap=g['tipgrad'] if mode=='pinch' else np.array([g['chaingrad'][n] for n in names])
      residual.extend(gaps/.001);jac.extend(dgap/.001)
      # Topological branch fixes the outer digits on opposite sides of the axis.
      for name,angle in [('index',100.),('pinky',-100.)]:
        p=row['local'][name][-1] if mode=='pinch' else g['chainpoint'][name]
        Jp=g['J'][name][-1] if mode=='pinch' else g['chainpointgrad'][name]
        v=p[:2]-row['x'][16:18];rho=max(np.linalg.norm(v),1e-12);n=v/rho
        target=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
        weight=2. if mode=='wrap' else .25
        residual.extend(weight*(n-target));jac.extend(weight*((np.eye(2)-np.outer(n,n))/rho)@(Jp[:2]-g['O']))
      if mode=='pinch':
        tips=np.array([row['local'][n][-1] for n in DIGITS]);Js=np.array([g['J'][n][-1] for n in DIGITS]);dp=np.zeros(18)
        for i in range(5):
          j=(i+1)%5;d=tips[i]-tips[j];dp+=(d/max(np.linalg.norm(d),1e-15))@(Js[i]-Js[j])
        residual.append(.08*row['tip_perimeter_m']/.02);jac.append(.08*dp/.02)
      # Weak movement penalty selects one posture without inventing new axes.
      residual.extend(.002*(row['x'][:16]-self.seed[:16]));jac.extend(.002*np.eye(18)[:16])
      r=np.array(residual);J=np.array(jac)
      return float(r@r),2*r@J

    def fun(self,z,mode):
      row,g=self.get(z,mode);value,grad=self.objective_data(row,g,mode)
      return value,grad*self.scale

    def constraint(self,z,mode):
      row,g=self.get(z,mode)
      # A 50um interior margin prevents numerical boundary drift past 0.2mm.
      value=(g['collision']+self.pen-.00005)/.001;J=g['collisiongrad']/.001
      if mode=='pinch':value=np.r_[value,g['axial']/.001];J=np.vstack([J,g['axialgrad']/.001])
      return value,J*self.scale

    def structured_seed(self):
      # Algebraic reach branch: folded rays extend backward toward axis_x≈-2mm.
      q=np.zeros(21)
      values=[(90.,-19.,75.),(90.,0.,24.),(90.,0.,28.),(105.,19.5,70.)]
      for i,(m,a,p) in enumerate(values):q[4*i:4*i+3]=np.deg2rad([m,a,p])
      q[16:]=[np.deg2rad(65.),np.deg2rad(40.),0.,np.deg2rad(70.),np.deg2rad(65.)]
      q=anatomy.project(q);return np.r_[Feedback._pack(q),-.002,0.]

    def run(self,out):
      out=Path(out);out.mkdir(parents=True,exist_ok=True);self.out=out
      if (out/'FK_ledger.jsonl').exists():raise RuntimeError('Existing ledger; continuation must preserve its budget')
      def dump(n,v):(out/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
      dump('semantic_program.json',self.program);dump('model_snapshot.json',self.model)
      guard=normal_cylinder_reach_guard(self.model,self.request['object']['circumference_m'],self.tol);dump('analytic_reach_guard.json',guard)
      # The audit is itself charged to this round's budget, including every probe.
      base=self.evaluate(self.seed);J=point_jacobians(self.model,base);errors=[]
      for j in range(18):
        x=self.seed.copy();step=1e-7 if j<16 else 1e-8;x[j]+=step
        changed=self.evaluate(x)
        error=max(float(np.max(abs((changed['local'][n]-base['local'][n])/step-J[n][:,:,j]))) for n in DIGITS)
        errors.append(error)
      audit={'FK_used':self.budget.used['FK'],'max_point_derivative_error':max(errors),'per_control_errors':errors,'pass':max(errors)<1e-6}
      dump('derivative_audit.json',audit)
      if not audit['pass']:raise RuntimeError('Analytic derivative audit failed; budget ledger retained')
      print('Derivative audit passed; FK used '+str(self.budget.used['FK']),flush=True)
      warm=self.structured_seed();summary=[]
      for mode,quota in self.program['schedule']['per_candidate_quotas'].items():
        self.current_mode=mode;self.phase_start=self.budget.used['FK'];self.quota=quota
        lo=self.lo.copy();hi=self.hi.copy()
        if mode in ('grasp','pinch'):
          hi[16]=min(hi[16],guard['axis_x_required_branches_m']['negative_max'])
          lo[16]=max(lo[16],guard['index_reach_requires_axis_x_at_least_m'])
        else:hi[16]=min(hi[16],.01)
        start=np.clip(warm,lo+1e-10,hi-1e-10)
        status=''
        try:
          opt=minimize(lambda z:self.fun(z,mode),start/self.scale,jac=True,method='SLSQP',bounds=list(zip(lo/self.scale,hi/self.scale)),
            constraints=[{'type':'ineq','fun':lambda z:self.constraint(z,mode)[0],'jac':lambda z:self.constraint(z,mode)[1]}],
            options={'maxiter':180,'ftol':1e-9,'disp':False})
          status=str(opt.message)
        except StopBudget as exc:status=str(exc)
        candidates=[]
        for key in self.phase_keys.get(mode,[]):
          row=self.cache[key];g=self.geometry(row);value,_=self.objective_data(row,g,mode)
          violation=float(max(0.,-g['collision'].min()-self.pen))
          if mode=='pinch':violation=max(violation,float(max(0.,-g['axial'].min())))
          candidates.append(((violation>1e-8,violation if violation>1e-8 else 0.,value),row))
        if not candidates:raise RuntimeError('No recorded candidate; ledger retained')
        _,row=min(candidates,key=lambda c:c[0]);warm=row['x'].copy();obs=self.observe(row,mode)
        # Replace inherited mixed axial/radial tip distance with the exact side observer.
        g=self.geometry(row)
        if mode=='pinch':
          gaps=g['tip'];eligible=np.array([self.span[0]<=-row['local'][n][-1,2]<=self.span[1] for n in DIGITS]);contacts=eligible&(gaps>=-self.pen)&(gaps<=self.tol)
          obs['contact_gaps_mm']=dict(zip(DIGITS,(gaps*1000).tolist()));obs['contact_count']=int(contacts.sum())
          obs['status']='sampled_geometric_proxy_pass' if contacts.all() and obs['max_penetration_mm']<=self.pen*1000+1e-7 else 'not_achieved'
        obs.update({'termination':status,'FK_evaluations_in_phase':self.budget.used['FK']-self.phase_start,
          'analytic_reach_bounds_applied':True,'derivative_method':'audited analytic serial-chain derivatives; no finite-difference pose search',
          'side_cap_binding_audit':'side-only contact objective and observation; caps count only in collisions'})
        result=self.export(row,mode,obs);result['target_contract']['id']=self.runlabel+'_'+mode
        result['render_hints'].pop('hide_inactive_placeholder',None);result['provenance'].pop('inactive_hand',None)
        dump(mode+'_result.json',result);dump(mode+'_assessment.json',obs);summary.append(obs)
        print(json.dumps({'mode':mode,'contacts':str(obs['contact_count'])+'/'+str(obs['required_contact_count']),'status':obs['status'],
          'penetration_mm':obs['max_penetration_mm'],'span_deg':obs['sampled_angular_span_deg'],'FK_used':self.budget.used['FK'],'stop':status}),flush=True)
      report={'schema':'naruto.hand_object_experiment.v2','summary':summary,'analytic_guard':guard,'budget':self.budget.report(),
        'derivative_audit':audit,'random_samples':0,'structural_candidates':3,'restarts':0,'elapsed_seconds':time.monotonic()-self.start,
        'anatomy_sha256':anatomy.SHA256,'new_round_authorization':'User: 那继续做对','physical_acceptance':False,
        'previous_500_ledger_preserved':True,'axis_and_diameter_unchanged':True,'postprocessing_new_FK':0}
      dump('report.json',report);return report
