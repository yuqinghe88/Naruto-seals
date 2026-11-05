"""Static inventory: discover source contracts WITHOUT executing case scripts."""
from assembly.paths import input_path
from pathlib import Path
import ast
import json
import re
import hashlib
from solvers.lbs_domain import INDEPENDENT
from solvers.p1_registry import ROOT,load


def source_record(path):
 text=input_path(path).read_text();tree=ast.parse(text);overrides=[];masks=[];functions=[];calls=set();imports=set()
 for node in ast.walk(tree):
  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):functions.append(node.name)
  if isinstance(node,ast.Call):
   name=node.func.id if isinstance(node.func,ast.Name) else (node.func.attr if isinstance(node.func,ast.Attribute) else '')
   calls.add(name)
  if isinstance(node,ast.ImportFrom):imports.add(node.module or '')
  if isinstance(node,ast.Import):imports.update(x.name for x in node.names)
  if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
   targets=node.targets if isinstance(node,ast.Assign) else [node.target]
   for target in targets:
    source=ast.get_source_segment(text,target) or ''
    if 'JOINT_LIMITS[' in source:overrides.append({'line':node.lineno,'target':source})
    if re.search(r'\b(active|CORE|active_indices|bounds|limits)\b',source):
     masks.append({'line':node.lineno,'target':source,'expression':(ast.get_source_segment(text,node.value) or '')[:320]})
 return {'file':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(input_path(path).read_bytes()).hexdigest(),'functions':sorted(set(functions)),'limit_overrides':overrides,'mask_or_bound_expressions':masks,'legacy_four_mm_literals':bool(re.search(r'(?<![\w.])(?:0)?\.004(?!\d)',text)),
         'two_axis_wrist_evidence':"'screen_normal_rotation'" in text or 'wrist_angles.ravel()' in text,
         'imports':sorted(imports),'calls':calls}


def run(output=None):
 old=json.loads((input_path(ROOT/'seal_registry.json')).read_text());registered={}
 for seal,entry in old['seals'].items():
  for row in entry.get('cases',[]):
   if row.get('script'):registered.setdefault(row['script'],[]).append({'seal':seal,'case':row['id']})
  if entry.get('solver_script'):registered.setdefault(entry['solver_script'],[]).append({'seal':seal,'entry':True})
 generic={'actual_state_transition.py','anchored_interlock_feedback.py','overlap_feedback.py','unweave_feedback.py','interlock_preserving_solver.py','fist_semantic_feedback.py','fist_protocol_compiler.py','local_action_candidates.py','mcp_opposition_loss.py','bird_manual_policy.py','bird_two_phase_interlock.py','bird_four_tip_docking.py','bird_palm_approach.py','gesture_feedback.py','nih_lbs_rig.py','planner_execution.py','solve_seal.py'}
 rows=[]
 paths=set(ROOT.glob('*.py'))|{ROOT/name for name in registered if (ROOT/name).is_file()}|{ROOT/name for name in ('solvers/dragon.py','solvers/lbs_domain.py','solvers/service.py','solvers/audit.py')}
 for p in sorted(paths):
  if p.name.startswith('test_') or p==ROOT/'solvers/audit.py':continue
  row=source_record(p)
  discovered=bool(row['calls']&{'minimize','least_squares','Feedback','_apply_q'})
  relative=str(p.relative_to(ROOT))
  if relative not in registered and p.name not in generic and p.parent!=ROOT/'solvers' and not discovered:continue
  row.pop('calls');row['registered_cases']=registered.get(relative,[])
  native=p.name in {'nih_lbs_rig.py','planner_execution.py','solve_seal.py'} or p.parent==ROOT/'solvers'
  row['disposition']='native_planner_path' if native else 'historical_not_automatically_executable_on_LBS'
  findings=[]
  if row['limit_overrides']:findings.append({'kind':'authority_override','severity':'blocked_for_native_execution','reason':'scripts attempt to overwrite sole ROM authority; limits must be removed in a migrated adapter'})
  if row['two_axis_wrist_evidence']:findings.append({'kind':'underused_wrist','reason':'historical two-axis wrist state omits third rotation; native runtime supports three world-frame rotation coordinates, clinical calibration unknown'})
  if row['legacy_four_mm_literals']:findings.append({'kind':'restricted_search_window','reason':'source contains 4mm constants; inspect recorded expressions, do not interpret them as an anatomical limit'})
  if row['mask_or_bound_expressions'] and not native:findings.append({'kind':'stage_mask_requires_review','reason':'source-local indices are recorded, not guessed into new LBS indices; compare against current explicit allowed_dofs and freezes'})
  row['findings']=findings;rows.append(row)
 report={'schema':'naruto.p1_solver_dof_audit.v1','method':'static AST/source evidence, no historical script imports/execution','authority':'anatomy_authority.json','native_dispatch':'production_planner.execute','native_independent_controls':INDEPENDENT,'stored_coordinates_per_two_hands':54,'independent_controls_per_two_hands':len(INDEPENDENT),'limits':'per-digit full authority; candidate trust steps are not anatomy limits','clinical_wrist_axes':'unknown; three declared world-frame rotation coordinates around fixed LBS pivot','inventory':rows,'summary':{'sources':len(rows),'historical_case_files':len(registered),'limit_override_sources':sum(bool(x['limit_overrides']) for x in rows),'two_axis_wrist_sources':sum(x['two_axis_wrist_evidence'] for x in rows),'four_mm_literal_sources':sum(x['legacy_four_mm_literals'] for x in rows)},'selection_policy':'only bound native adapters automatically selected; history requires explicit migration','missing_registered_source_paths':[name for name in registered if not (ROOT/name).is_file()],'dynamic_audit':'every planner result reports requested, active, frozen-dependent and unused authoritative coordinates per stage','caveat':'a static expression or unused coordinate alone does not prove a stage should release it'}
 if output:Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 return report
if __name__=='__main__':
 out=ROOT/'dist/p1_solver_audit.json';report=run(out);print(json.dumps(report['summary']))
