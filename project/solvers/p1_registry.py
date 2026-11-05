"""Registered solver capabilities; historical scripts are data, never runpy."""
from assembly.paths import input_path
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]

def load():return json.loads((input_path(ROOT/'planning/p1_solver_registry.json')).read_text())
def binding(identifier):
 rows=load()['adapters']
 if identifier not in rows:raise ValueError('unregistered_solver: '+identifier)
 return rows[identifier]

def legacy_trial(controller,action,budget,samples=4):
 """Transactional legacy adapter. Every FK/sensing request shares one meter.

 This does not port the old two-axis wrist/capsule model to current LBS.
 """
 import copy
 from actual_state_transition import step
 snapshot=copy.deepcopy(controller.__dict__);apply=controller._apply_q;measure=controller.measure
 def counted_apply(*args,**kwargs):budget.charge('legacy_fk');return apply(*args,**kwargs)
 def counted_measure(*args,**kwargs):budget.charge('legacy_sense');return measure(*args,**kwargs)
 controller._apply_q=counted_apply;controller.measure=counted_measure
 try:
  candidate,report=step(controller,action,samples=samples,allow_over_tolerance=False)
  return {'model':'legacy_spring_feedback','candidate_packed':candidate.tolist(),'report':report,'promoted_to_LBS':False,'physical_acceptance':False}
 finally:
  controller.__dict__.clear();controller.__dict__.update(snapshot)
