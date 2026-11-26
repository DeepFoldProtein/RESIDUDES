# The closest public pose: 6S9K, 14-3-3 gamma with the caspase-2 pS139 peptide.
reinitialize
bg_color white
set ray_opaque_background, 1
set antialias, 2
set ray_trace_mode, 1
set ray_trace_color, 0x404040
set ray_shadows, 1
set ambient_occlusion_mode, 1
set specular, 0.25
set depth_cue, 1
set fog_start, 0.55
set cartoon_fancy_helices, 1
set stick_radius, 0.22
python
import numpy as _np
def face(on, partner, zoom_on="", buffer=4):
    p = cmd.get_coords(on)
    r = cmd.get_coords(f"({partner}) within 10 of ({on})")
    if p is None or r is None:
        cmd.orient(zoom_on or on)
        return
    z = p.mean(0) - r.mean(0)
    z /= _np.linalg.norm(z)
    x = _np.linalg.svd(p - p.mean(0))[2][0]
    x -= x.dot(z) * z
    if _np.linalg.norm(x) < 1e-6:
        x = _np.cross(z, [1.0, 0, 0] if abs(z[0]) < 0.9 else [0, 1.0, 0])
    x /= _np.linalg.norm(x)
    v = list(cmd.get_view())
    v[0:9] = _np.array([x, _np.cross(z, x), z]).T.flatten().tolist()
    cmd.set_view(v)
    cmd.zoom(zoom_on or f"({on}) or ({partner})", float(buffer))
cmd.extend("face", face)
python end
load <WORK>/targets/H1388/dl/pdb/6S9K.pdb, s9k
remove solvent or hydro or resn CFH
hide everything
show cartoon, chain A
color 0xDADADA, chain A
show sticks, chain A and resi 50+57+132+133 and not name N+C+O
color 0x0072B2, chain A and resi 50+57+132+133
show cartoon, chain B
set cartoon_loop_radius, 0.45
show sticks, chain B and resn SEP and not name N+C+O
color 0xD55E00, chain B
face chain B, chain A, chain B, 9
png figs/pocket_6s9k.png, width=1800, height=1350, dpi=300, ray=1
