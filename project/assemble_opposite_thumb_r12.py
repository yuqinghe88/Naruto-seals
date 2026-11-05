"""Direct geometric thumb assembly from a fixed support and the shaft surface."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

import anatomy_authority as anatomy
import naruto
from animation_model.geometry import segment_box, segment_closest
from animation_model.rigs.spring import Assembler

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'runs/hand_object_d25_opposite_tip_sum_r11'
OUT = ROOT / 'runs/hand_object_d25_manual_assembly_r12'
DIGITS = ('thumb', 'index', 'middle', 'ring', 'pinky')


def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def radial_gap(a, b, center, radius):
    """Lower bound on finite shaft clearance from its infinite side wall."""
    p = a[:2] - center
    v = b[:2] - a[:2]
    squared = v @ v
    t = float(np.clip(-(p @ v) / squared, 0., 1.)) if squared > 1e-24 else 0.
    return float(np.linalg.norm(p + t * v) - radius)


def capsule_gap(a, b, c, d, radius_sum):
    p, q, _, _ = segment_closest(a, b, c, d)
    return float(np.linalg.norm(p - q) - radius_sum)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    base = json.loads((SOURCE / 'result.json').read_text())
    model = json.loads((SOURCE / 'model_snapshot.json').read_text())
    spec = model['hands'][0]['fingers']['thumb']
    shaft = base['brush_shaft']
    hand = base['hands'][0]
    fingers = hand['fingers']
    axis = np.array(shaft['axis_direction_world_m'])
    center = np.array(shaft['axis_point_world_m'][:2])
    root = np.array(spec['root_in_palm_m'])
    l0, l1, l2 = spec['segment_lengths_m']
    thumb_radius = spec['radius_m']
    shaft_radius = shaft['radius_m']

    # Set the first link along the palmar normal. Recover the existing CMC
    # coordinates algebraically and retain the specified PS coupling.
    frame = np.array(spec['finger_chirality_matrix'])
    canonical = frame.T @ axis
    yaw = np.arctan2(canonical[1], canonical[0])
    elevation = np.arctan2(-canonical[2], np.linalg.norm(canonical[:2]))
    prior = naruto.THUMB_REST_PRIOR
    fe = yaw + np.deg2rad(prior['radial_spread_deg'])
    aa = -elevation - np.deg2rad(prior['outward_elevation_deg'])
    ps = anatomy.D_PS * fe + anatomy.E_PS * aa
    bending = (frame @ naruto.thumb_cmc_transform(fe, aa, ps)[:3, :3])[:, 1]
    assert abs(bending @ axis) < 1e-12

    # Take the near intersection of the thumb's vertical bending plane
    # with the cylinder inflated by the fingertip radius.
    offset = root[:2] - center
    projection = offset @ bending[:2]
    contact_radius = shaft_radius + thumb_radius
    discriminant = projection**2 - offset @ offset + contact_radius**2
    lateral = -projection - np.sqrt(discriminant)
    assert lateral > 0
    index = fingers['index']
    index_depth = -index['points_world_m'][-1][2]
    depth = index_depth - index['radius_m'] - thumb_radius - .002
    assert shaft['span_m'][0] < depth < shaft['span_m'][1]
    target = root + depth * axis + lateral * bending

    # The outward elbow of the two-link triangle keeps the phalanges
    # on the outside of the cylinder until the fingertip touches.
    axial = depth - l0
    cosine = (axial**2 + lateral**2 - l1**2 - l2**2) / (2 * l1 * l2)
    assert -1 <= cosine <= 1
    ip = np.arccos(cosine)
    mcp = np.arctan2(lateral, axial) - np.arctan2(l2 * np.sin(ip), l1 + l2 * np.cos(ip))
    mcp_point = root + l0 * axis
    ip_point = mcp_point + l1 * (np.cos(mcp) * axis + np.sin(mcp) * bending)
    points = np.array([root, mcp_point, ip_point, target])
    assert np.allclose(np.linalg.norm(np.diff(points, axis=0), axis=1), [l0, l1, l2], atol=1e-12)

    # Replay the assembled angles once, without limit projection, to verify
    # that the original hinges and the same bone lengths reproduce the points.
    q = np.array(base['q_left'])
    q[16:] = [fe, aa, ps, mcp, ip]
    backend = Assembler.__new__(Assembler)
    backend.model = model
    replay = backend.local(0, q)
    assert np.max(abs(replay['thumb'] - points)) < 1e-12
    assert all(np.max(abs(replay[name] - fingers[name]['points_world_m'])) < 1e-12 for name in DIGITS[1:])

    all_points = {name: np.array(fingers[name]['points_world_m']) for name in DIGITS[1:]}
    all_points['thumb'] = points
    rod_gaps = {}
    for name in DIGITS:
        chain = all_points[name]
        radius = fingers[name]['radius_m']
        rod_gaps[name] = min(radial_gap(a, b, center, shaft_radius + radius) for a, b in zip(chain, chain[1:]))
    assert min(rod_gaps.values()) >= -1e-10
    thumb_pair_gaps = {}
    for name in DIGITS[1:]:
        chain = all_points[name]
        thumb_pair_gaps[name] = min(capsule_gap(a, b, c, d, thumb_radius + fingers[name]['radius_m'])
                                    for a, b in zip(points, points[1:])
                                    for c, d in zip(chain, chain[1:]))
    assert min(thumb_pair_gaps.values()) >= -1e-10
    assert capsule_gap(*points[:2], *points[2:], 2 * thumb_radius) >= -1e-10

    half = np.array(hand['palm_size_m']) / 2
    # Existing embedded attachment may pass inside the palm. Its exposed
    # capsule starts once its center is one radius beyond the palmar face.
    release_z = -half[2] - thumb_radius
    t = (release_z - mcp_point[2]) / (ip_point[2] - mcp_point[2])
    assert 0 < t < 1
    release = mcp_point + t * (ip_point - mcp_point)
    assert np.all(np.abs(release[:2]) < half[:2])
    palm_gap = min(segment_box(a, b, np.zeros(3), np.eye(3), half)[0] - thumb_radius
                   for a, b in ((release, ip_point), (ip_point, target)))
    wrist_gap = min(capsule_gap(a, b, *np.array(hand['wrist_world_m']),
                               thumb_radius + model['hands'][0]['wrist']['radius_m'])
                    for a, b in zip(points, points[1:]))
    assert min(palm_gap, wrist_gap) >= -1e-10

    tips = np.array([all_points[name][-1] for name in DIGITS])
    radials = tips[:, :2] - center
    radii = np.linalg.norm(radials, axis=1)
    unit = radials / radii[:, None]
    support = unit[1:].sum(axis=0)
    support /= np.linalg.norm(support)
    side_dot = float(unit[0] @ support)
    assert side_dot < 0
    gaps = radii - shaft_radius - np.array([fingers[name]['radius_m'] for name in DIGITS])
    assert abs(gaps[0]) < 1e-12
    assert all(-.0002 <= gap <= .0005 for gap in gaps)
    assert all(shaft['span_m'][0] <= -p[2] <= shaft['span_m'][1] for p in tips)
    contact = target.copy()
    contact[:2] -= thumb_radius * unit[0]
    assert abs(np.linalg.norm(contact[:2] - center) - shaft_radius) < 1e-12

    limits = anatomy.check(q)
    assert limits['status'] == 'fail' and not limits['coupling_violations']
    exceeded = []
    for i, name in enumerate(anatomy.THUMB_KEYS, start=16):
        lo, hi = anatomy.LIMITS_RAD[i]
        excess = max(float(lo - q[i]), float(q[i] - hi), 0.)
        if excess > 1e-9:
            exceeded.append({'joint': name, 'angle_deg': float(np.rad2deg(q[i])),
                             'limit_deg': np.rad2deg([lo, hi]).tolist(),
                             'excess_deg': float(np.rad2deg(excess))})
    measurement = {
        'status': 'manual_geometric_contact_rom_exceeded',
        'relation': 'pinch', 'contact_count': 5, 'required_contact_count': 5,
        'contact_gaps_mm': dict(zip(DIGITS, (gaps * 1000).tolist())),
        'thumb_contact_point_world_m': contact.tolist(), 'thumb_contact_axial_mm': float(depth * 1000),
        'thumb_to_four_resultant_deg': float(np.rad2deg(np.arccos(side_dot))),
        'opposite_half_plane': True,
        'thumb_to_four_distance_sum_mm': float(np.linalg.norm(tips[0] - tips[1:], axis=1).sum() * 1000),
        'thumb_joint_angles_deg': np.rad2deg(q[16:]).tolist(), 'exceeded_limits': exceeded,
        'anatomy': limits, 'max_penetration_mm': 0.,
        'rod_capsule_clearance_lower_bound_mm': {name: gap * 1000 for name, gap in rod_gaps.items()},
        'thumb_other_finger_capsule_gaps_mm': {name: gap * 1000 for name, gap in thumb_pair_gaps.items()},
        'thumb_palm_clearance_mm': palm_gap * 1000, 'thumb_wrist_clearance_mm': wrist_gap * 1000,
        'optimizer_evaluations': 0, 'fk_replays': 1,
        'physical_acceptance': False,
    }
    pose = copy.deepcopy(base)
    pose['status'] = measurement['status']
    pose['hands'][0]['fingers']['thumb']['points_world_m'] = points.tolist()
    pose['hands'][0]['control_q_rad'] = q.tolist()
    pose['hands'][0]['control_validation'] = limits
    pose['hands'][0]['role'] = 'direct_geometric_assembly_rom_exceeded'
    pose['q_left'] = q.tolist()
    pose['positions_m'] = [p for name in DIGITS for p in all_points[name].tolist()]
    pose['metrics'].update({'accepted': False, 'penetration_m': 0., 'contact_error_m': float(abs(gaps).max())})
    pose['target_contract'] = {'id': 'd25_manual_assembly_r12', 'pose_variant': 'pinch',
                               'current_locked_target_description': 'direct geometric assembly, fixed four, opposite side contact',
                               'declared_relations': {'assembly': 'explicit shaft tangency and two-link triangle',
                                                      'thumb_root_ray': 'palmar normal', 'fixed': 'four fingers and rod',
                                                      'allow_rom_excess': True}, 'measured': measurement}
    pose['provenance'].update({'geometry': 'direct assembly verified by one unprojected FK replay',
                               'optimizer_evaluations': 0, 'fk_replays': 1, 'source': 'R11 fixed four contacts'})
    write('result.json', pose)
    write('checks.json', measurement)
    write('model_snapshot.json', model)
    source = Path(__file__)
    write('change_registry.json', [{'path': source.name, 'before_sha256': None,
                                   'after_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                   'reason': 'direct five-finger geometric assembly and limit audit'}])
    print(json.dumps({k: measurement[k] for k in ['contact_count','contact_gaps_mm','thumb_to_four_resultant_deg',
                     'thumb_contact_axial_mm','exceeded_limits','thumb_other_finger_capsule_gaps_mm',
                     'thumb_to_four_distance_sum_mm','optimizer_evaluations','fk_replays']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
