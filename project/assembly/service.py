"""Control-to-FK frame assembly, with one caller-owned budget and rig."""
import copy
import numpy as np
from assembly.contracts import Artifact
from animation_model.body import attach_caster


def assemble(controls,rig,assembler,budget):
    controls.require('trajectory_controls',rig['rig_id'],rig['model_sha256'],rig['authority_sha256'])
    budget.require_continuation(controls.provenance.get('budget'))
    from pattern_visual_trial import screen
    payload=controls.payload;routes=[];arm_calls=0
    for route in payload['routes']:
        rows=[]
        for row in route['controls']:
            before=assembler.used;raw=assembler.assemble({k:np.array(v) for k,v in row['pose'].items()})
            budget.charge('FK',assembler.used-before);arm_calls+=2
            rows.append({'fraction':row['fraction'],'result':attach_caster(raw,payload['caster_policy'])})
        a,b=route['id'].split('->');edge=route['edge_policy']
        held=[x['source'] for x in edge['proposal']['analysis']['common_relations']]
        checks=screen(rows,held,payload['policy'])
        checks['partial_order_hold']='unknown; missing local region/contact policies not inferred'
        routes.append({'id':route['id'],'name':route['name'],'frames':rows,'summary':checks,'workspace_mode':edge['mode'],
            'keyposes':[{'name':a,'sample_index':0},{'name':'lookahead_transition','sample_index':8},{'name':b,'sample_index':16}],
            'display_notice':'Modular replay | human-rejected Hare: diagnostic only | holds/physics UNKNOWN' if controls.provenance['human_rejections'] else 'Modular assembly | candidate semantics/physics UNKNOWN'})
    policy=copy.deepcopy(payload['caster_policy']);policy['display'].update(fps=payload['policy']['fps'],render_every_nth_sample=1,
        hold_internal_seals=payload['policy']['hold_internal_seals'])
    result={'schema':'naruto.eleven_lookahead_sequence.v1','continuous_sequence':True,'order':payload['order'],'names':payload['names'],
        'policy':policy,'routes':routes,'physical_acceptance':'unknown','full_semantic_acceptance':'unknown',
        'visual_acceptance':'rejected_endpoint_present' if controls.provenance['human_rejections'] else 'needs_review'}
    return Artifact('assembled_frames',rig['rig_id'],rig['model_sha256'],rig['authority_sha256'],result,
        {'budget':budget.report(),'closed_form_arm_IK_calls':arm_calls,'human_rejections':controls.provenance['human_rejections'],
         'physical_acceptance':'unknown','continuous_path':'unknown'})


def maintain_interaction_group(binding, initial, model, authority, propose, assembler, budget):
    """Run caller's existing proposal backend with one shared FK meter.

    No relation is solved independently and no counter is reset. Explicit
    geometry transforms require their own versioned dispatcher/rebinding.
    """
    from interaction_group_runtime import execute_group
    original_limit = assembler.budget
    def measured(state, attempt):
        remaining = budget.limits['FK'] - budget.used['FK']
        if remaining <= 0:
            return None
        assembler.budget = min(original_limit, assembler.used + remaining)
        before = assembler.used
        try:
            return propose(state, attempt)
        finally:
            budget.charge('FK', assembler.used - before)
    try:
        result = execute_group(binding, initial, model, authority, measured)
        if budget.used['FK'] >= budget.limits['FK'] and result['stop'] == 'completed':
            result['stop'] = 'shared_FK_budget_exhausted'
        result['budget'] = budget.report()
        return result
    finally:
        assembler.budget = original_limit
