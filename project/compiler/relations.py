"""Finite local relation-graph matching and versioned transition motifs.

Matches are descriptions, never observations. No mirror/role swap, implicit
joint freeze, unknown wildcard, inverse edge or physical acceptance is inferred.
"""
from __future__ import annotations
from assembly.paths import input_path
import copy
import hashlib
import json
import re
from pathlib import Path

from assembly.paths import ROOT
CATALOG = 'planning/transition_motifs_20261031.json'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def unknown(value):
    return value is None or (isinstance(value, str) and
        (value.lower().startswith(('unknown', 'unavailable', 'unsupported'))))


def unknown_fields(value, path='spec'):
    if unknown(value): return [path]
    if isinstance(value, dict):
        return [p for k,v in value.items() for p in unknown_fields(v,path+'.'+k)]
    if isinstance(value, list):
        return [p for i,v in enumerate(value) for p in unknown_fields(v,f'{path}[{i}]')]
    return []


def canonical(kind, spec):
    result = copy.deepcopy(spec)
    # Only an unqualified symmetric touch may exchange its two arguments.
    # Direction, landing/order and per-anchor fields are never discarded.
    if kind == 'contact' and isinstance(result, dict) and result.get('kind') == 'touch':
        allowed = {'a','b','kind','contact_tolerance_m','penetration_tolerance_m'}
        if set(result) <= allowed and 'a' in result and 'b' in result:
            result['a'],result['b'] = sorted([result['a'],result['b']],key=fingerprint)
    return result


def anchors(value):
    """References only: no inferred region for legacy finger-level tokens."""
    found = []
    def walk(v):
        if isinstance(v,dict):
            if v.get('hand') in ('left','right') and 'structure' in v:
                found.append(copy.deepcopy(v)); return
            for child in v.values(): walk(child)
        elif isinstance(v,list):
            for child in v: walk(child)
        elif isinstance(v,str):
            m = re.fullmatch(r'([LR])_(thumb|index|middle|ring|pinky|palm|wrist)(?:_(tip))?',v)
            if m:
                found.append({'hand':'left' if m[1]=='L' else 'right','structure':m[2],
                    'region':{'type':'tip','id':'tip'} if m[3] else None,
                    'legacy_reference':v})
    walk(value)
    return list({fingerprint(x):x for x in found}.values())


def relations(node):
    target=node['target'];rel=target.get('relations',{});result=[]
    def add(kind,spec,path):
        c=canonical(kind,spec)
        result.append({'id':fingerprint({'kind':kind,'spec':c}), 'kind':kind,
            'spec':copy.deepcopy(spec),'canonical_spec':c,'evidence_path':path,
            'anchors':anchors(spec),'unresolved_fields':unknown_fields(spec),
            'actual_status':'unknown'})
    for kind in ('interlock','thumb_interlock','palm_contact','max_contact'):
        if rel.get(kind):add(kind,rel[kind],'target.relations.'+kind)
    for i,spec in enumerate(rel.get('contacts',[])):
        add('contact',spec,f'target.relations.contacts[{i}]')
    for i,spec in enumerate(target.get('additional_relations',[])):
        add(spec.get('kind','unsupported'),spec,f'target.additional_relations[{i}]')
    return result


def leaves(value,path='spec'):
    if isinstance(value,dict):
        return {p:w for k,v in value.items() for p,w in leaves(v,path+'.'+k).items()}
    if isinstance(value,list):
        return {p:w for i,v in enumerate(value) for p,w in leaves(v,f'{path}[{i}]').items()}
    return {path:value}


def control_profile(source,target):
    out={};sf=source['target'].get('factors',{});tf=target['target'].get('factors',{})
    for side in ('left','right'):
        keys=[f'{side}.{d}.{j}' for d in ('index','middle','ring','pinky') for j in ('MCP','spread','PIP')]
        keys += [f'{side}.thumb.{j}' for j in ('CMC_FE','CMC_AA','MCP','IP')]
        valid=source.get('factor_status')!='reference_only' and target.get('factor_status')!='reference_only'
        known=[k for k in keys if valid and k in sf and k in tf]
        out[side]={'known_independent_controls':len(known),'expected_independent_controls':len(keys),
            'missing_controls':[k for k in keys if k not in known],
            'reference_change_rad':sum(abs(tf[k]-sf[k]) for k in known),
            'unchanged_preferences':[k for k in known if abs(tf[k]-sf[k])<=1e-10],
            'freeze_constraints':[]}
    return out


def interlock_fragments(node):
    """Local corresponding-pair order references, not invented contact specs."""
    spec=node['target'].get('relations',{}).get('interlock') or {}
    word=spec.get('handed_order') or [];found=[]
    for i in range(len(word)-1):
        a,b=word[i:i+2]
        ma=re.fullmatch(r'([LR])_(thumb|index|middle|ring|pinky)',a)
        mb=re.fullmatch(r'([LR])_(thumb|index|middle|ring|pinky)',b)
        if not ma or not mb or ma[1]==mb[1] or ma[2]!=mb[2]:continue
        context={k:copy.deepcopy(spec[k]) for k in ('class','frame','landing_side','landing_region','contact_policy')
                 if k in spec and not unknown(spec[k])}
        found.append({'id':fingerprint({'kind':'corresponding_pair_order','tokens':[a,b]}),
            'kind':'corresponding_pair_order','tokens':[a,b],'word_offset':i,
            'anchors':anchors([a,b]),'declared_context':context,
            'contact_requirement':'unknown_unless_explicitly_bound_to_this_pair',
            'region':'unknown_for_legacy_finger_level_order','freeze_constraints':[],
            'actual_status':'unknown','source_relation':copy.deepcopy(spec)})
    return found


def common_fragments(nodes):
    groups=[interlock_fragments(n) for n in nodes];common=[]
    for fragment in groups[0]:
        matches=[next((x for x in g if x['id']==fragment['id']),None) for g in groups]
        if any(x is None for x in matches):continue
        contexts=[x['declared_context'] for x in matches]
        conflicts={k for k in set().union(*(set(c) for c in contexts))
                   if len({fingerprint(c[k]) for c in contexts if k in c})>1}
        if conflicts:continue
        common.append({'id':fragment['id'],'tokens':fragment['tokens'],'anchors':fragment['anchors'],
            'matched_offsets':[x['word_offset'] for x in matches],
            'contexts':contexts,'contact_and_region_status':'unknown',
            'freeze_constraints':[],'scope':'shared local order candidate; full group relation not inherited'})
    return common


def analyze(source,target):
    sr,tr=relations(source),relations(target);used=set();common=[]
    for item in sr:
        match=next((i for i,x in enumerate(tr) if i not in used and x['id']==item['id']),None)
        if match is None:continue
        other=tr[match];used.add(match)
        missing=list(dict.fromkeys(item['unresolved_fields']+other['unresolved_fields']))
        if item['kind']=='max_contact':missing+=['coverage_measure_and_acceptance_threshold']
        if item['kind'] in ('interlock','thumb_interlock'):
            for field in ('anchors','frame','contact_policy'):
                if field not in item['spec']:missing.append('spec.'+field)
        common.append({'source':item,'target':other,'match':'equal_declared_contract',
            'unresolved_fields':missing,'hold':'candidate_requires_actual_check',
            'freeze_constraints':[]})
    shared={x['source']['id'] for x in common}
    release=[x for x in sr if x['id'] not in shared]
    acquire=[x for i,x in enumerate(tr) if i not in used]
    partial=[]
    for a in release:
        for b in acquire:
            if a['kind']!=b['kind']:continue
            la,lb=leaves(a['canonical_spec']),leaves(b['canonical_spec'])
            same={k:v for k,v in la.items() if k in lb and not unknown(v) and not unknown(lb[k]) and v==lb[k]}
            changed={k:{'source':la[k],'target':lb[k]} for k in la.keys()&lb.keys()
                     if not unknown(la[k]) and not unknown(lb[k]) and la[k]!=lb[k]}
            if not same:continue
            partial.append({'source_id':a['id'],'target_id':b['id'],'kind':a['kind'],
                'common_declared_fields':same,'changed_declared_fields':changed,
                'missing_or_unknown_fields':sorted((la.keys()^lb.keys())|{k for k,v in {**la,**lb}.items() if unknown(v)}),
                'role':'local_similarity_hint_only_not_a_hold_or_proven_logical_conflict'})
    exclusions=[]
    if target['target'].get('explicit_no_interlock'):
        exclusions=[{'source_id':x['id'],'reason':'target_explicit_no_interlock'} for x in sr if x['kind']=='interlock']
    return {'schema':'naruto.relation_pattern_analysis.v1',
        'source_target_hashes':[source['target_version_sha256'],target['target_version_sha256']],
        'source_relations':sr,'target_relations':tr,'common_relations':common,
        'release_candidates':release,'acquire_candidates':acquire,'local_similarities':partial,
        'local_order_fragments':common_fragments([source,target]),
        'explicit_conflicts':exclusions,'controls':control_profile(source,target),
        'hand_role_or_mirror_mapping':'identity_only; no implicit swaps',
        'actual_source_status':'unknown','unchanged_controls_are_frozen':False,
        'target_omission_is_prohibition':False,'physical_acceptance':'unknown'}


def propose(source,target,catalog=None):
    catalog=catalog or json.loads((input_path(ROOT/CATALOG)).read_text());a=analyze(source,target)
    by_id={x['id']:x for x in catalog['motifs']};result=[]
    support=None
    if all(not x['missing_controls'] for x in a['controls'].values()):
        support=min(a['controls'],key=lambda s:(a['controls'][s]['reference_change_rad'],s))
        other='right' if support=='left' else 'left'
        if a['controls'][support]['reference_change_rad'] > catalog['support_activity_ratio']*a['controls'][other]['reference_change_rad']:
            support=None
    ids=[]
    if a['common_relations'] and not a['explicit_conflicts']:ids.append('preserve_common')
    if support is not None and not a['common_relations']:ids.append('support_hand')
    ids.append('release_rebuild')
    for mid in ids:
        definition=by_id[mid]
        keep=a['common_relations'] if mid=='preserve_common' else []
        keepids={x['source']['id'] for x in keep}
        payload={'motif_id':mid,'motif_version':definition['version'],
            'catalog_sha256':fingerprint(catalog),'source_target_hashes':a['source_target_hashes'],
            'actual_contract_hashes':[fingerprint(source['target']),fingerprint(target['target'])],
            'keep_source_relation_ids':sorted(keepids),
            'keep_target_relation_ids':sorted(x['target']['id'] for x in keep),
            'release_relation_ids':[x['id'] for x in a['source_relations'] if x['id'] not in keepids],
            'support_hand':support if mid=='support_hand' else None,
            'stages':copy.deepcopy(definition['stages']), 'freeze_constraints':[],
            'validation_tier':'unvalidated_candidate','physical_acceptance':'unknown',
            'role':'proposed_control_schedule_not_achieved_relation_effect'}
        payload['candidate_id']=fingerprint(payload);result.append(payload)
    return {'analysis':a,'candidates':result,'candidate_cap':catalog['candidate_cap']}


def validate_candidate(source,target,candidate):
    current=propose(source,target)
    match=next((x for x in current['candidates'] if x==candidate),None)
    if match is None:raise ValueError('Stale, edited, reversed or unregistered motif candidate')
    return current['analysis']


def reusable_record(candidate,observations,model_sha256,authority_sha256):
    """Evidence envelope, not automatic generalization or inverse registration."""
    return {'schema':'naruto.motif_instance.v1','candidate':copy.deepcopy(candidate),
        'model_sha256':model_sha256,'authority_sha256':authority_sha256,
        'source_target_hashes':candidate['source_target_hashes'],
        'validation':copy.deepcopy(observations),'validation_tier':'screened_visual_candidate',
        'human_visual_acceptance':'pending','static_physics':'unknown','continuous_path':'unknown',
        'automatic_promotion':False,'inverse_evidence_inherited':False,
        'reuse':'reinstantiate and recheck current versions; exact instance evidence only'}


def validate_instance(record,source,target,model_sha256,authority_sha256,required_tier='visual_screen'):
    validate_candidate(source,target,record['candidate'])
    if record['model_sha256'] != model_sha256:
        raise ValueError('Model changed; motif instance must be re-instantiated and rechecked')
    if record['authority_sha256'] != authority_sha256:
        raise ValueError('Authority changed; motif instance must be re-instantiated and rechecked')
    if required_tier != 'visual_screen' or record['validation'].get('status') != 'screened_candidate':
        raise ValueError('Stored instance does not certify the requested validation tier')
    return {'status':'eligible_for_reinstantiation','runtime_recheck_required':True,
            'physical_acceptance':'unknown','inverse_evidence_inherited':False}
