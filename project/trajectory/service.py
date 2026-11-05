"""Sample controls only; FK and rendering belong to downstream stages."""
import copy
import numpy as np
from assembly.contracts import Artifact,ContractError,RunBudget
from assembly.io import read
from assembly.paths import resolve
from trajectory.sequence import SequencePose


def sample(plan,poses,rig,assembler):
    plan.require('semantic_plan',rig['rig_id'],rig['model_sha256'],rig['authority_sha256'])
    poses.require('pose_set',rig['rig_id'],rig['model_sha256'],rig['authority_sha256'])
    if rig['rig_id']!='legacy_spring_feedback':raise ContractError('Trajectory backend not migrated; no implicit rig conversion')
    if 'budget' not in poses.provenance:raise ContractError('Missing inherited pose budget')
    inherited=RunBudget.resume(poses.provenance['budget'])
    if 'budget' in plan.provenance:inherited.require_continuation(plan.provenance['budget'])
    from eleven_lookahead_sequence import edge_policy
    config=read(resolve(rig['catalog']['policies']['trajectory']));sequence=plan.payload
    nodes=[sequence['nodes'][s] for s in sequence['order']]
    values=[{k:np.array(v) for k,v in poses.payload['poses'][sid].items()} for sid in sequence['order']]
    edges=[edge_policy(a,b) for a,b in zip(nodes,nodes[1:])];path=SequencePose(values,nodes,edges,assembler,config)
    routes=[]
    for i,(a,b,edge) in enumerate(zip(nodes,nodes[1:],edges)):
        controls=[]
        for t in np.linspace(i,i+1,config['samples_per_edge']):
            p=path(float(t));controls.append({'fraction':float(t-i),'pose':{k:v.tolist() for k,v in p.items()}})
        routes.append({'id':a['id']+'->'+b['id'],'name':a['name']+'→'+b['name'],'controls':controls,'edge_policy':edge})
    return Artifact('trajectory_controls',rig['rig_id'],rig['model_sha256'],rig['authority_sha256'],
        {'order':sequence['order'],'names':sequence['names'],'routes':routes,'windows':path.windows,
         'policy':config,'caster_policy':read(resolve(read(resolve(rig['catalog']['policies']['endpoint']))['caster_policy']))},
        {'sampling_performs_FK':False,'physical_acceptance':'unknown','budget':poses.provenance['budget'],
         'human_rejections':poses.provenance['human_rejections']})
