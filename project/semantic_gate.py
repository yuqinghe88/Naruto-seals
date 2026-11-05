"""L0 gate for explicit interlock requirements and executable action bindings."""
import naruto


def validate(seal,actions,requirements=None):
    diagnostics=[]; requirements=requirements or {}
    def add(code,path,detail):diagnostics.append({'code':code,'path':path,'detail':detail})
    relation=seal.get('interlock') or {}
    if requirements.get('interlock_segment_anchor_required') and relation and not relation.get('anchors'):
        add('INTERLOCK_REGION_UNSPECIFIED','interlock.anchors','Explicit requirement needs segment anchors; finger-level pairs do not identify segments.')
    for i,a in enumerate(actions):
        p=a.get('parameters') or {}; family=a.get('family'); path=f'actions[{i}]'
        if family=='Thumb' and p.get('to') is None:
            add('NULL_ACTION_TARGET',path+'.parameters.to','No clear-constraint execution binding is declared.')
        if family=='RelHand':
            controls=p.get('controls')
            if isinstance(controls,dict) and 'finger_shape_goals' in controls:
                declared=controls['finger_shape_goals']
                if any(declared.get(side,{}).get('index')!=seal['semantics'][side]['index'] for side in ('left','right')):
                    add('ACTION_CONTROL_TARGET_CONFLICT',path+'.parameters.controls.finger_shape_goals','Index shape conflicts with the gesture target.')
                if controls.get('binding_status')!='verified':
                    add('ACTION_CONTROL_CAPABILITY_UNKNOWN',path+'.parameters.controls.binding_status','Runtime binding and its supported controls are not verified.')
            if isinstance(controls,str) and 'straight-index aim controls' in controls:
                shapes=[seal['semantics'][side]['index'] for side in ('left','right')]
                if any(v!='straight' for v in shapes):
                    add('ACTION_CONTROL_CAPABILITY_UNKNOWN',path+'.parameters.controls','Straight-index wording conflicts with target labels; actual supported domain is not machine declared.')
    return {'status':'unknown' if diagnostics else 'pass_for_checked_rules_only','diagnostics':diagnostics,
            'scope':'L0 selected checks, not full semantic or physical validation'}
