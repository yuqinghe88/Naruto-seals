"""Compatibility entry delegating to the same planner service as every seal."""
from .service import run,variant_program

def solve(variant='contact',*,initial=None,budget=200,output_dir=None):
 if variant not in ('contact','overlap'):raise ValueError('unknown recorded Dragon variant')
 # Variant remains an explicit unbound requested contract; canonical target is unchanged;
 # the historical overlap/contact recipes are never imported or replayed.
 report,out=run('dragon',program=variant_program('dragon',variant),initial=initial,budget=budget,output_dir=output_dir)
 return out/'result.json'
