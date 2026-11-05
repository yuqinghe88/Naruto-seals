"""Versioned anchor-driven relation groups. Compilation performs zero FK."""
import copy,math
from assembly.io import fingerprint
from assembly.contracts import ContractError
DIGITS={'thumb','index','middle','ring','pinky'}

def anchor(a,model):
    if not isinstance(a,dict) or set(a)-{'hand','structure','region'}:raise ContractError('unsupported anchor fields')
    if a.get('hand') not in ('left','right'):raise ContractError('explicit hand required')
    h=next((h for h in model['hands']if h['side']==a['hand']),None)
    if h is None or a.get('structure')not in set(h['fingers'])|{'palm'}:raise ContractError('anchor absent from rig')
    if a.get('region')not in ('tip','chain','loop','MCP','CMC','center','edge'):raise ContractError('unbound anchor region')
    return copy.deepcopy(a)

def compile_group(spec,model,authority,capabilities=()):
    allowed={'schema','id','relations','freezes','actions','allowed_controls','budget'}
    if set(spec)-allowed or spec.get('schema')!='naruto.interaction_group.v1':raise ContractError('unsupported group schema/fields')
    if not spec.get('relations'):raise ContractError('nonempty relations required')
    ids=set();out=copy.deepcopy(spec)
    for r in out['relations']:
        if r.get('id')in ids or not isinstance(r.get('id'),str):raise ContractError('unique relation ids required')
        ids.add(r['id']);kind=r.get('kind')
        fields={'id','kind','a','b','contact_tolerance_m','penetration_tolerance_m'}if kind=='Touch'else {'id','kind','actor','guide','closure','evidence','sliding'}if kind=='Threaded'else set()
        if not fields or set(r)-fields:raise ContractError('unsupported relation or fields')
        if kind=='Touch':
            for k in ('a','b'):
                r[k]=anchor(r[k],model)
                if r[k]['region']!='tip':raise ContractError('only endpoint-sphere touch bound')
            for k in ('contact_tolerance_m','penetration_tolerance_m'):
                if isinstance(r.get(k),bool)or not isinstance(r.get(k),(int,float))or not math.isfinite(r[k])or r[k]<0:raise ContractError('finite tolerances required')
        else:
            for k in ('actor','guide'):r[k]=anchor(r[k],model)
            if r['actor']['region']!='chain'or r['guide']['region']!='loop':raise ContractError('thread chain and loop required')
            if r.get('closure')!='virtual_tip_to_root'or r.get('evidence')!='static_centerline_crossing'or r.get('sliding')is not True:raise ContractError('only explicit sliding static virtual aperture bound; directed insertion needs path observer')
    for f in out.get('freezes',[]):
        if set(f)!={'anchor','mode','tolerance'}:raise ContractError('explicit freeze fields required')
        f['anchor']=anchor(f['anchor'],model)
        if f['anchor']['region']!='chain':raise ContractError('freeze currently binds explicit whole chains only')
        if f['mode']not in ('world_points','shape','joint_angles'):raise ContractError('unknown freeze mode')
        if not isinstance(f['tolerance'],(int,float))or isinstance(f['tolerance'],bool)or not math.isfinite(f['tolerance'])or f['tolerance']<0:raise ContractError('invalid freeze tolerance')
    controls=out.get('allowed_controls',[])
    if len(controls)!=len(set(controls))or any(c not in capabilities for c in controls):raise ContractError('undeclared rig control; no invented axes')
    for f in out.get('freezes',[]):
        if f['mode']=='joint_angles'and any(c.startswith(f['anchor']['hand']+'.'+f['anchor']['structure']+'.')for c in controls):raise ContractError('frozen joint control requested')
    for a in out.get('actions',[]):
        if set(a)!={'kind','axis','pivot','frame','angle_deg','backend'}:raise ContractError('explicit transform action contract required')
        if a['kind']not in ('RotateAboutPalmEdge','RebindFingerFrame','MirrorAcrossPalmPlane','ReorientFinger'):raise ContractError('unbound action')
        if a['frame']not in ('world','palm_local')or len(a['axis'])!=3 or not all(math.isfinite(x)for x in a['axis'])or sum(x*x for x in a['axis'])<1e-12:raise ContractError('explicit finite nonzero axis/normal required')
        anchor(a['pivot'],model)
        if not math.isfinite(a['angle_deg']):raise ContractError('finite transform angle required')
        if a['backend']not in capabilities:raise ContractError('action backend unavailable; not assumed achieved')
    if not isinstance(out.get('budget'),int)or isinstance(out['budget'],bool)or not 1<=out['budget']<=200:raise ContractError('explicit shared proposal budget 1..200 required')
    return {'schema':'naruto.bound_interaction_group.v1','spec':out,'model_sha256':fingerprint(model),'authority_sha256':fingerprint(authority),
        'FK_execution':False,'status':'bound_not_achieved','coupling_source':'anatomy_authority.json','continuous_path':'unknown','physical_acceptance':False}
