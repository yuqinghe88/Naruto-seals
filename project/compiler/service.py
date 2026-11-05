"""Semantic-only stage adapter; no FK or rendering execution."""
from assembly.paths import input_path
import hashlib
from assembly.contracts import Artifact,ContractError
from assembly.paths import resolve
from assembly.io import read


def compile_request(request,rig,budget):
    if rig['rig_id']!='legacy_spring_feedback':raise ContractError('Sequence semantic adapter currently targets explicit spring scope')
    # Read versioned contract material, not a gesture builder that also creates
    # geometry. Reuse the existing selection/planner algorithms, without FK.
    from eleven_semantic_sequence import choose_order
    from tree_transition_compiler import infer
    from p2_directed_candidates import infer_edge
    def material(name):
        item=rig['catalog'][name];path=resolve(item['path'])
        if hashlib.sha256(input_path(path).read_bytes()).hexdigest()!=item['sha256']:raise ContractError('Semantic material changed: '+name)
        return read(path)
    if request.get('motion_or_FK_execution') is not False or request.get('model')!=rig['rig_id']:
        raise ContractError('Semantic compile request must explicitly disable FK and match rig')
    if request.get('visit_each_scope_node_once') is not True or request.get('return_to_start') is not False:
        raise ContractError('Only the declared open visit-sequence request is currently bound')
    tree=material('tree');nodes=material('semantic_nodes')['nodes'];known={x['id']:x for x in material('intermediates')['candidates']}
    limit=budget.limits['abstract']-budget.used['abstract']
    order,trace,used,status=choose_order(tree,nodes,request['start'],request['ranking_policy'],limit)
    transitions=[];compiled=0
    for a,b in zip(order,order[1:]):
        if used>=limit:break
        edge=infer_edge(nodes[a],nodes[b],known)
        leg=infer(tree,nodes[a],nodes[b],remaining_expansions=limit-used,graph_edge_support=edge)
        work=leg['planner_result']['expansions'];used+=work;compiled+=work
        leg['semantic_edge']=edge;transitions.append(leg)
        if leg['status']!='conditional_symbolic_route':break
    complete=status=='complete' and len(transitions)==len(nodes)-1 and all(x['status']=='conditional_symbolic_route' for x in transitions)
    result={'schema':'naruto.eleven_semantic_sequence.v1','request':dict(request),
        'status':'conditional_semantic_sequence' if complete else 'incomplete_under_budget',
        'order':order,'names':[nodes[s]['name'] for s in order],'nodes':nodes,'transitions':transitions,'selection_trace':trace,
        'budget':{'selection_scores':used-compiled,'planner_expansions':compiled,'used_total_abstract_work':used,'maximum':limit},
        'summary':{'FK_evaluations':0,'motion_executions':0,'physical_acceptance':'unknown','globally_optimal_order':False}}
    budget.charge('abstract',used)
    return Artifact('semantic_plan',rig['rig_id'],rig['model_sha256'],rig['authority_sha256'],result,
        {'adapter':'existing finite compiler','FK_execution':False,'budget':budget.report(),'physical_acceptance':'unknown'})


def import_saved(path,rig):
    path=resolve(path);result=read(path)
    wrapped=Artifact.from_dict(result) if result.get('schema')=='naruto.stage_artifact.v1' else None
    if wrapped is not None:
        wrapped.require('semantic_plan',rig['rig_id'],rig['model_sha256'],rig['authority_sha256']);result=wrapped.payload
    if result.get('schema')!='naruto.eleven_semantic_sequence.v1':raise ContractError('Not a saved semantic sequence')
    for sid,node in result['nodes'].items():
        expected=rig['catalog']['targets'].get(sid)
        if expected is None or expected['target_version']!=node['target_version_sha256']:
            raise ContractError('Saved target version differs from current catalog: '+sid)
    if wrapped is not None:return wrapped
    return Artifact('semantic_plan',rig['rig_id'],rig['model_sha256'],rig['authority_sha256'],result,
        {'source':str(path),'source_sha256':hashlib.sha256(input_path(path).read_bytes()).hexdigest(),
         'source_work':result['budget'],'new_abstract_work':0,'reuse_is_physical_acceptance':False})


def compile_interaction_group(request, model, authority, capabilities=()):
    """Anchor-driven group binding; no FK or optimizer execution."""
    from compiler.interaction_elements import compile_group
    return compile_group(request, model, authority, capabilities)


def compile_hand_object_relation(request, model, authority):
    """Bind object axis, contact anchors and relation intent; zero FK."""
    from compiler.hand_object import compile_hand_object
    return compile_hand_object(request, model, authority)
