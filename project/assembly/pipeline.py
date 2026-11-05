"""Unified stage orchestration; individual services remain independently usable."""
from assembly.contracts import RunBudget,ContractError
from assembly.artifacts import ArtifactStore
from assembly.paths import ROOT,resolve,output_path
from assembly.io import read
from animation_model.catalog import load
from animation_model.rigs.spring import Assembler


def run(saved_plan,rig_id,output,allow_rejected=False):
    from compiler.service import import_saved
    from pose.service import initialize,ensure_allowed
    from trajectory.service import sample
    from assembly.service import assemble
    destination=output_path(output)
    if destination.exists() and any(destination.iterdir()):raise ContractError('Run output must be new or empty; no silent overwrite')
    rig=load(rig_id);plan=import_saved(saved_plan,rig);budget=RunBudget.resume(plan.provenance.get('budget'))
    if rig_id!='legacy_spring_feedback':raise ContractError('NIH sequence pipeline not migrated; no spring fallback')
    ensure_allowed(plan,rig,allow_rejected)
    config=read(ROOT/rig['catalog']['policies']['endpoint'])
    assembler=Assembler(rig['model'],budget.limits['FK']-budget.used['FK'],amendments=read(ROOT/config['thumb_amendments']))
    poses=initialize(plan,rig,assembler,budget,allow_rejected)
    controls=sample(plan,poses,rig,assembler);frames=assemble(controls,rig,assembler,budget)
    store=ArtifactStore(output)
    for name,item in [('01_plan',plan),('02_poses',poses),('03_controls',controls),('04_frames',frames)]:store.save(name,item)
    store.finish(budget,human_rejections=frames.provenance['human_rejections'],purpose='migration_parity_replay',physical_acceptance='unknown')
    return frames,store.root
