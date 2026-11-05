"""One project path authority; computation modules do not own output folders."""
from pathlib import Path
import json
from functools import lru_cache
ROOT=Path(__file__).resolve().parents[1]


def resolve(path):
    path=Path(path)
    return path if path.is_absolute() else ROOT/path


@lru_cache(maxsize=4)
def _input_aliases(root,stamp):
    manifest=root/'assets/input_paths.json'
    return json.loads(manifest.read_text())['aliases'] if manifest.exists() else {}


def input_path(path):
    """Resolve a historical read, never an output or an implicit latest result.

    Registered aliases always select the immutable snapshot, even if an old
    experiment creates a new file at its former dist path. Outside-project
    paths keep their ordinary meaning. No links or copies are left in dist.
    """
    path=Path(path)
    try:key=str((path if path.is_absolute() else ROOT/path).resolve().relative_to(ROOT))
    except ValueError:return path
    manifest=ROOT/'assets/input_paths.json'
    aliases=_input_aliases(ROOT,manifest.stat().st_mtime_ns if manifest.exists() else None)
    entry=aliases.get(key)
    if entry is None:return path
    target=ROOT/entry['path']
    if not target.is_file():raise FileNotFoundError('Registered input snapshot missing: '+entry['path'])
    return target


def output_path(path):
    """Keep new staged writes separate from pinned input aliases and snapshots."""
    path=resolve(path)
    manifest=ROOT/'assets/input_paths.json'
    aliases=_input_aliases(ROOT,manifest.stat().st_mtime_ns if manifest.exists() else None)
    try:key=str(path.resolve().relative_to(ROOT))
    except ValueError:return path
    if key in aliases or key=='assets/inputs' or key.startswith('assets/inputs/'):
        raise ValueError('Pinned input is read-only; choose a new run output path')
    return path
