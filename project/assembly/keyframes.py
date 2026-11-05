"""One explicit static candidate recipe; no automatic target/tree registration."""
import hashlib
from assembly.paths import ROOT,output_path
from assembly.io import read,dump


def generate(target,output,budget=180):
    if target not in ('zhe','zhe-forced','zhe-twist','zhe-contact'):raise ValueError('Unregistered candidate recipe')
    destination=output_path(output)
    if destination.exists()and any(destination.iterdir()):raise ValueError('Use a new candidate output directory')
    path={'zhe':'planning/zhe_reference_candidate_20261031.json','zhe-forced':'planning/zhe_forced_bend_candidate_20261031.json','zhe-twist':'planning/zhe_pip_roll_candidate_20261031.json','zhe-contact':'planning/zhe_contact_feedback_20261031.json'}[target]
    spec=read(ROOT/path)
    image=spec['reference_image']
    if hashlib.sha256((ROOT/image['path']).read_bytes()).hexdigest()!=image['sha256']:raise ValueError('Reference changed')
    if target=='zhe':
        from pose.recipes.zhe import solve
    elif target=='zhe-forced':
        from pose.recipes.zhe_forced import solve
    elif target=='zhe-twist':
        from pose.recipes.zhe_twist import solve
    else:
        from pose.solvers.forced_contact import solve
    result,report=solve(spec,budget)
    if target=='zhe-contact':
        from assets.manager import load
        from animation_model.contact_snapshot import inspect
        from animation_model.render.contact_feedback import render
        ghost=report.pop('zero_roll_reference');inspection=inspect(result,load('legacy_spring_feedback')['model'])
        dump(destination/'result.json',result);dump(destination/'zero_roll_display_reference.json',ghost)
        dump(destination/'collision_observation.json',inspection)
        report['diagnostic_media']=render(result,ghost,inspection,destination)
        dump(destination/'report.json',report)
        return report,destination
    dump(destination/'result.json',result);dump(destination/'report.json',report)
    from animation_model.render.keyframe import render_candidate
    title='Zhe | reference keyframe candidate | physics NOT accepted'if target=='zhe'else'Zhe | forced-bend trial | threading NOT achieved'
    notice=None if target=='zhe'else'Three tip pairs retained | Ring threading: FAIL | PIP lateral / twist: UNBOUND | Physics UNKNOWN'
    if target=='zhe-twist':
        title='Zhe | experimental PIP roll | ring threading proxy '+('PASS'if report['threading_geometry_proxy_pass']else'FAIL')
        notice='PIP roll is an experimental extension | Base bone lengths retained | Opening/path/physics NOT certified'
    render_candidate(result,destination,title,notice=notice)
    return report,destination
