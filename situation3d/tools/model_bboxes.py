# -*- coding: utf-8 -*-
"""Build a collision-bbox manifest from the normalized GLB assets (stdlib only)."""
import json, os, struct, sys
import numpy as np

COMP = {5120: ('b', 1), 5121: ('B', 1), 5122: ('h', 2), 5123: ('H', 2),
        5125: ('I', 4), 5126: ('f', 4)}
NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def load_glb(path):
    with open(path, 'rb') as f:
        magic, ver, length = struct.unpack('<III', f.read(12))
        chunks = []
        while f.tell() < length:
            clen, ctype = struct.unpack('<II', f.read(8))
            chunks.append((ctype, f.read(clen)))
    js = json.loads(chunks[0][1])
    binary = chunks[1][1] if len(chunks) > 1 else b''
    return js, binary


def accessor(js, binary, idx):
    a = js['accessors'][idx]
    bv = js['bufferViews'][a['bufferView']]
    off = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    comp = COMP[a['componentType']]
    n = NCOMP[a['type']]
    stride = bv.get('byteStride', comp[1] * n)
    out = np.empty((a['count'], n), dtype=np.float64)
    for k in range(a['count']):
        out[k] = struct.unpack_from('<' + comp[0] * n, binary, off + k * stride)
    return out


def quat_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def node_matrix(n):
    if 'matrix' in n:
        return np.array(n['matrix'], dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    t = n.get('translation', [0, 0, 0])
    r = n.get('rotation', [0, 0, 0, 1])
    s = n.get('scale', [1, 1, 1])
    m[:3, :3] = quat_matrix(r) * np.array(s)[None, :]
    m[:3, 3] = t
    return m


def world_points(js, binary):
    out = []
    scene = js.get('scenes', [{}])[js.get('scene', 0)]
    roots = scene.get('nodes', [])

    def walk(idx, parent):
        node = js['nodes'][idx]
        mat = parent @ node_matrix(node)
        if 'mesh' in node:
            for prim in js['meshes'][node['mesh']]['primitives']:
                attrs = prim.get('attributes') or {}
                if 'POSITION' not in attrs:
                    continue
                v = accessor(js, binary, attrs['POSITION'])
                vh = np.c_[v, np.ones(len(v))]
                out.append((mat @ vh.T).T[:, :3])
        for child in node.get('children', []):
            walk(child, mat)

    for root in roots:
        walk(root, np.eye(4))
    return np.vstack(out) if out else np.empty((0, 3))


def main(models_dir, out_engine, out_web):
    models = {}
    for fn in sorted(os.listdir(models_dir)):
        if not fn.startswith('type_') or not fn.endswith('.glb') or '_gunfx' in fn:
            continue
        try:
            js, binary = load_glb(os.path.join(models_dir, fn))
            if any('draco' in str(e).lower() for e in js.get('extensionsUsed', [])):
                print('SKIP draco', fn)
                continue
            pts = world_points(js, binary)
            if not len(pts):
                continue
            mn, mx = pts.min(axis=0), pts.max(axis=0)
            dims = mx - mn
            models[fn] = {
                'size_m': [round(float(dims[0]), 2), round(float(dims[1]), 2),
                           round(float(dims[2]), 2)],
                'length_m': round(float(max(dims[0], dims[2])), 2),
                'width_m': round(float(min(dims[0], dims[2])), 2),
                'height_m': round(float(dims[1]), 2),
                'min_m': [round(float(x), 2) for x in mn],
                'max_m': [round(float(x), 2) for x in mx],
            }
            print(fn, models[fn]['length_m'], models[fn]['width_m'], models[fn]['height_m'])
        except Exception as exc:
            print('SKIP', fn, repr(exc))
    doc = {'schema': 'situation3d.model_bboxes.v1', 'models': models}
    for path in (out_engine, out_web):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write('\n')
    print('wrote', len(models), 'models')


if __name__ == '__main__':
    root = sys.argv[1]
    main(os.path.join(root, 'web/assets/models'),
         os.path.join(root, 'engine/data/models/bboxes.json'),
         os.path.join(root, 'web/assets/models/bboxes.json'))
