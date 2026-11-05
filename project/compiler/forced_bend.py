"""Contract for constraint-induced bending; not a new anatomical DOF or force law."""
import copy


def bind(contract,extension=None):
    if contract.get('kind')!='forced_bend':raise ValueError('Expected forced_bend relation')
    if contract.get('affected_joint')!='PIP':raise ValueError('Only explicit PIP directional bending is bound')
    if contract.get('guide',{}).get('closure')!='semi_closed':raise ValueError('Explicit semi-closed guide required')
    actor=contract['actor'];guide=contract['guide']['anchor']
    if actor['hand']==guide['hand']:raise ValueError('This recipe requires an opposite-hand guide')
    if actor['structure']!='ring'or guide['structure']!='middle':raise ValueError('This binding is ring through opposite middle')
    if actor.get('region')!={'type':'segment','id':'segment_2'}:raise ValueError('Explicit ring segment_2 required')
    regions=contract['guide'].get('boundary_anchors',[])
    if {x.get('region',{}).get('id')for x in regions}!={'segment_1','segment_2','segment_3'}:
        raise ValueError('Middle loop needs its three explicit segment anchors')
    result={'schema':'naruto.forced_bend_binding.v1','contract':copy.deepcopy(contract),
        'resolved_joint':'PIP (second joint counted MCP, PIP, DIP; not segment_2)',
        'supported_controls':['ring.MCP_flex','ring.MCP_abd','ring.PIP_flex','coupled_DIP'],
        'direction_binding':'PIP-to-DIP segment aimed across opposite middle-loop plane',
        'loop_binding':'planar centerline polygon with an explicitly virtual open-mouth edge',
        'PIP_lateral_bend':'unsupported_by_current_authority',
        'PIP_axial_twist':'unsupported_by_current_authority',
        'passive_force_equilibrium':'not_bound','continuous_threading':'not_verified',
        'overall':'partial_binding_not_full_semantic_acceptance'}
    if extension is not None:
        if extension.get('schema')!='naruto.forced_pip_extension.v1':raise ValueError('Explicit authorized extension required')
        result['PIP_axial_twist']='experimental_hinge_plane_roll_bound; anatomical_ROM_unknown'
        result['supported_controls'].append('ring.PIP_roll_explicit_forced_bend_only')
        result['extension_rig']=extension['rig_id']
    return result
