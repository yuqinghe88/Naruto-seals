"""One offline service entry for planner-backed seal and explicit-program runs."""
from assembly.paths import input_path
from pathlib import Path
import hashlib
import json
from production_planner import execute
from semantic_goal_planning import compile_runtime_seal,relaxed_lbs_result

ROOT=Path(__file__).resolve().parents[1]


def run_hand_object_relation(request, output_dir):
 """Explicit fixed-rig hand--object experiment with a shared persisted budget."""
 from solvers.hand_object import HandObjectRun
 return HandObjectRun(request).run(output_dir)

def variant_program(seal,variant):
 program=compile_runtime_seal(seal);registry=json.loads((input_path(ROOT/'seal_registry.json')).read_text());record=registry['seals'][seal].get('variants',{}).get(variant)
 if record is None:raise ValueError('unknown historical variant '+str(variant))
 program['requested_historical_variant']={'seal':seal,'variant':variant,'record':record}
 program['stages'].append({'id':'verify_requested_variant','allowed_dofs':[],
  'goals':[{'kind':'relation','contract':{'kind':'historical_variant_migration','spec':{'seal':seal,'variant':variant,'record':record}}}],
  'attempts':[{'id':'variant_binding','solver':'lbs.local_feedback'}]})
 return program


def run(seal=None,program=None,initial=None,budget=200,output_dir=None):
 program=compile_runtime_seal(seal) if program is None else program
 initial=relaxed_lbs_result() if initial is None else initial
 try:result=execute(program,initial,budget=budget)
 except ValueError as exc:result={'schema':'naruto.p1_execution_result.v1','status':'protocol_error','failure':{'status':'unknown','reason':str(exc)},'budget':{'limit':budget,'used':0,'scope':'protocol rejected before execution'},'final_result':None,'physical_acceptance':False,'continuous_path':'unknown'}
 out=Path(output_dir) if output_dir else ROOT/'dist/planner_runs'/(seal or 'explicit_program')
 out.mkdir(parents=True,exist_ok=True)
 for name,data in [('program.json',program),('initial.json',initial),('report.json',result),('result.json',result['final_result'])]:
  (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 return result,out
