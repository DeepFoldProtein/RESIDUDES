# The first results: where each predictor puts the LARP1 peptide, on 14-3-3 gamma (af3 + pS7's receptor shown).
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
load <WORK>/targets/H1388/results/A1B1/sep/af3/seed_20/sample_0/model.cif, af3_pS
load <WORK>/targets/H1388/results/A1B1/sep/pt2/seed_5/sample_0/model.cif, pt2
load <WORK>/targets/H1388/results/A1B1/sep/bt2/seed_1/sample_0/model.cif, bt2
load <WORK>/targets/H1388/results/A1B1/base/af3/seed_5/sample_0/model.cif, af3_noP
load <WORK>/targets/H1388/results/A1B1/sep/of3/seed_6/sample_0/model.cif, of3
remove hydro
super pt2 and chain B and name CA, af3_pS and chain B and name CA
super bt2 and chain B and name CA, af3_pS and chain B and name CA
super af3_noP and chain B and name CA, af3_pS and chain B and name CA
super of3 and chain B and name CA, af3_pS and chain B and name CA
hide everything
show cartoon, af3_pS and chain B
color 0xDADADA, af3_pS and chain B
show sticks, af3_pS and chain B and resi 49+56+131+132 and not name N+C+O
color 0x0072B2, af3_pS and chain B and resi 49+56+131+132
show cartoon, chain A
set cartoon_loop_radius, 0.45
show sticks, chain A and resi 7 and not name N+C+O
color 0xD55E00, af3_pS and chain A
color 0x009E73, pt2 and chain A
color 0xCC79A7, bt2 and chain A
color 0xE69F00, af3_noP and chain A
color 0x56B4E9, of3 and chain A
face chain A, af3_pS and chain B, chain A, 9
png figs/pep_split.png, width=1800, height=1350, dpi=300, ray=1
