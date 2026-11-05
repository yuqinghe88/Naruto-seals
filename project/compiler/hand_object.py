"""Hand--object morphemes. Compile contracts without FK or pose search."""
import copy, math
from assembly.contracts import ContractError
from assembly.io import fingerprint

DIGITS=('thumb','index','middle','ring','pinky')

def compile_hand_object(request,model,authority):
    if request.get('schema')!='naruto.hand_object_request.v1':raise ContractError('hand-object request version')
    if request.get('hand') not in ('left','right'):raise ContractError('explicit hand')
    C=request['object']['circumference_m']
    if not math.isfinite(C) or C<=0:raise ContractError('positive circumference')
    if request['relation']['theta_deg']!=0:raise ContractError('this adapter binds theta=0 only')
    modes=request['modes']
    if modes!=['wrap','grasp','pinch']:raise ContractError('fixed three relation families required')
    normal=model['palm_surfaces']['palmar']['normal_in_palm']
    return {
      'schema':'naruto.hand_object_program.v1','request':copy.deepcopy(request),
      'model_sha256':fingerprint(model),'anatomy_sha256':fingerprint(authority),
      'morphemes':{
        'ObjectAxis':{'object':'cylinder','radius_m':C/(2*math.pi),'circumference_m':C,
          'axis_in_palm':normal,'axial_span_m':request['object']['axial_span_m'],
          'finite_length_assumption':'100mm shaft; near cap 12mm from palm mid-plane, on palmar side; no cap contacts count as side grasp'},
        'AlignAxis':{'theta_deg':0,'definition':'acos(abs(palm_normal dot cylinder_axis))','enforcement':'fixed equality; no rotational search'},
        'OffsetAxis':{'frame':'palm_local','parameters':['axis_x_m','axis_y_m'],'initial_m':[.020,0.]},
        'TouchObject':{'anchors':['tip','visible_segment'],'surface':'lateral_cylinder',
          'contact_tolerance_m':request['contact_tolerance_m'],'penetration_tolerance_m':request['penetration_tolerance_m']},
        'AroundAxis':{'normal_frame':'cylinder_cross_section','angular_threshold_rad':math.pi,
          'observer':'existing semantic_relations.wrap sampled angular span; not continuous material coverage or force closure'},
        'AxialOrder':{'policy':'axial staggering allowed, finger identities preserved; no same-height fingertip constraint'},
        'TipClosure':{'cycle':list(DIGITS),'metric':'3D adjacent tip-center perimeter; soft radii retained; not set equal to physical cylinder circumference'},
        'KeepAnatomy':{'bone_lengths':'fixed','roots':'fixed','palm_arch':'existing fixed installation; no added DOF',
          'limits':'anatomy_authority.json','coupling':['DIP=2/3 PIP','CMC ps=D*fe+E*aa']},
        'NoPenetration':{'pairs':['visible finger capsules / cylinder including caps','finger / finger','finger / palm','finger / wrist'],
          'embedded_attachment_exclusion':'reuse existing gesture_feedback attachment trimming'}},
      'operators':[
        {'family':'Flex','binding':'existing Feedback._pack/_unpack + anatomy projection','scope':'MCP and PIP; DIP coupled'},
        {'family':'Spread','binding':'existing MCP abduction','scope':'per digit authority bounds'},
        {'family':'Thumb','binding':'naruto.thumb_segment_rotations','scope':'CMC_FE/AA and MCP/IP; CMC_PS coupled'},
        {'family':'RelObject','binding':'new explicit object-frame translation, analogous to existing RelHand','scope':'two transverse offsets; theta locked'},
        {'family':'ObserveWrap','binding':'semantic_relations.wrap','scope':'existing static capsule angular proxy; topology remains unknown'}],
      'contracts':{
        'wrap':{'contact_digits':['middle','ring'],'anchors':'visible chain','around_axis_min_rad':math.pi,
          'acceptance':'sampled angular proxy only; full topological enclosure unknown'},
        'grasp':{'contact_digits':list(DIGITS),'anchors':'visible chain','around_axis_min_rad':math.pi,
          'acceptance':'five-digit contact and opposing angular span; mechanical stability unknown'},
        'pinch':{'contact_digits':list(DIGITS),'anchors':'tip','objective':'minimize adjacent 3D fingertip perimeter while all five side contacts hold',
          'acceptance':'five tips contact, no interpenetration; global minimum B unknown'}},
      'schedule':{'structural_candidates':3,'random_samples':0,'restarts':0,'shared_FK_budget':request['budget'],
        'per_candidate_quotas':{'wrap':240,'grasp':130,'pinch':129},'seed_evaluations':1,
        'stopping':['local least-squares convergence','candidate quota','total 500 FK','120 seconds wall clock'],
        'counting':'all unique FK states, including finite-difference Jacobian perturbations and seed; cached sensing/rendering adds no FK'},
      'status':'compiled_not_achieved','FK_evaluations':0}

def normal_cylinder_reach_guard(model,C,tolerance):
    """A necessary reach bound, independent of joint-space sampling.

    Four-finger XY chains lie in wedges |dy|<=tan(abd_max)*|dx|.
    The two outer MCP roots and thumb reach therefore bound axis_x.
    Excluding the whole real line certifies the five-contact contract impossible
    on this fixed rig, without claiming impossibility for human hands generally.
    """
    import numpy as np
    h=model['hands'][0];a=math.radians(20);R=C/(2*math.pi)
    i,p,t=(h['fingers'][n] for n in ('index','pinky','thumb'))
    separation=abs(i['root_in_palm_m'][1]-p['root_in_palm_m'][1])
    di=R+i['radius_m']+tolerance;dp=R+p['radius_m']+tolerance
    needed=(separation-(di+dp)/math.cos(a))/(2*math.tan(a))
    rootx=i['root_in_palm_m'][0]
    # Index MCP is limited to <=90deg, so its first segment has nonnegative X.
    # The last two segments cannot move X backward by more than their lengths.
    neg_lower=rootx-sum(i['segment_lengths_m'][1:])-di
    thumb_upper=t['root_in_palm_m'][0]+sum(t['segment_lengths_m'])+R+t['radius_m']+tolerance
    impossible=bool(needed>0 and neg_lower>rootx-needed and thumb_upper<rootx+needed)
    return {'kind':'theta0_five_digit_reach_necessary_condition','five_contact_impossible_on_this_rig':impossible,
      'axis_x_required_branches_m':{'negative_max':rootx-needed,'positive_min':rootx+needed},
      'index_reach_requires_axis_x_at_least_m':neg_lower,
      'thumb_reach_requires_axis_x_at_most_m':thumb_upper,
      'contact_tolerance_included_m':tolerance,'outer_root_separation_m':separation,
      'proof_scope':'fixed current rig; index MCP<=90deg, abduction<=20deg, finite phalanx lengths; any visible finger segment or tip; axial staggering does not alter this XY obstruction',
      'source':['anatomy_authority.json','gesture_pipeline.build_semantic_seal','triangle inequality and abduction wedge bound'],
      'note':'minimizing squared residual cannot override this exclusion; diagnostic closest candidates retain failed contracts'}
