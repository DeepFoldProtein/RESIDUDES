# residudes scenes: written by `residudes slides --final`; ray-traced with `pymol -cq scenes.pml` from report/
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
set_color plddt_vh, [0.000, 0.325, 0.839]
set_color plddt_h, [0.396, 0.796, 0.953]
set_color plddt_l, [1.000, 0.859, 0.075]
set_color plddt_vl, [1.000, 0.490, 0.271]
load <WORK>/targets/H1388/results/A1B1/sep/af3/seed_20/sample_0/model.cif, M1
load <WORK>/targets/H1388/results/A1B1/sep/pt2/seed_5/sample_0/model.cif, M2
load <WORK>/targets/H1388/results/A1B1/sep/af3/seed_9/sample_0/model.cif, M3
load <WORK>/targets/H1388/results/A1B1/sep/pt2/seed_10/sample_0/model.cif, M4
load <WORK>/targets/H1388/results/A1B1/sep/af3/seed_7/sample_0/model.cif, M5
load <WORK>/targets/H1388/results/A1B1/sep/af3/seed_14/sample_2/model.cif, C10
load <WORK>/targets/H1388/results/A1B1/sep/pt2/seed_4/sample_0/model.cif, C6
load <WORK>/targets/H1388/results/A1B1/sep/af3/seed_15/sample_3/model.cif, C2
load <WORK>/targets/H1388/results/A1B1/sep/of3/seed_6/sample_4/model.cif, C7
load <WORK>/targets/H1388/results/A1B1/base/af3/seed_2/sample_2/model.cif, C1
load <WORK>/targets/H1388/results/A1B1/base/af3/seed_5/sample_3/model.cif, C4
load <WORK>/targets/H1388/results/A1B1/sep/of3/seed_1/sample_2/model.cif, C0
load <WORK>/targets/H1388/results/A1B1/sep/pt2/seed_2/sample_0/model.cif, C9
remove solvent
remove hydro
hide everything
show cartoon
super M2 and name CA, M1 and name CA
super M3 and name CA, M1 and name CA
super M4 and name CA, M1 and name CA
super M5 and name CA, M1 and name CA
super C10 and name CA, M1 and name CA
super C6 and name CA, M1 and name CA
super C2 and name CA, M1 and name CA
super C7 and name CA, M1 and name CA
super C1 and name CA, M1 and name CA
super C4 and name CA, M1 and name CA
super C0 and name CA, M1 and name CA
super C9 and name CA, M1 and name CA
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
face M1 and chain A, M1 and not chain A, M1, 0
disable all
enable M1
util.cbc M1
color 0xD55E00, M1 and chain A
show spheres, M1 and chain A and not hydro
set sphere_scale, 0.45, M1 and chain A
set cartoon_transparency, 0.45, M1 and not chain A
png figs/M1_chains.png, width=1800, height=1350, dpi=300, ray=1
color plddt_vl, M1
color plddt_l, M1 and b > 50
color plddt_h, M1 and b > 70
color plddt_vh, M1 and b > 90
png figs/M1_plddt.png, width=1800, height=1350, dpi=300, ray=1
disable all
enable M2
util.cbc M2
color 0xD55E00, M2 and chain A
show spheres, M2 and chain A and not hydro
set sphere_scale, 0.45, M2 and chain A
set cartoon_transparency, 0.45, M2 and not chain A
png figs/M2_chains.png, width=1800, height=1350, dpi=300, ray=1
color plddt_vl, M2
color plddt_l, M2 and b > 50
color plddt_h, M2 and b > 70
color plddt_vh, M2 and b > 90
png figs/M2_plddt.png, width=1800, height=1350, dpi=300, ray=1
disable all
enable M3
util.cbc M3
color 0xD55E00, M3 and chain A
show spheres, M3 and chain A and not hydro
set sphere_scale, 0.45, M3 and chain A
set cartoon_transparency, 0.45, M3 and not chain A
png figs/M3_chains.png, width=1800, height=1350, dpi=300, ray=1
color plddt_vl, M3
color plddt_l, M3 and b > 50
color plddt_h, M3 and b > 70
color plddt_vh, M3 and b > 90
png figs/M3_plddt.png, width=1800, height=1350, dpi=300, ray=1
disable all
enable M4
util.cbc M4
color 0xD55E00, M4 and chain A
show spheres, M4 and chain A and not hydro
set sphere_scale, 0.45, M4 and chain A
set cartoon_transparency, 0.45, M4 and not chain A
png figs/M4_chains.png, width=1800, height=1350, dpi=300, ray=1
color plddt_vl, M4
color plddt_l, M4 and b > 50
color plddt_h, M4 and b > 70
color plddt_vh, M4 and b > 90
png figs/M4_plddt.png, width=1800, height=1350, dpi=300, ray=1
disable all
enable M5
util.cbc M5
color 0xD55E00, M5 and chain A
show spheres, M5 and chain A and not hydro
set sphere_scale, 0.45, M5 and chain A
set cartoon_transparency, 0.45, M5 and not chain A
png figs/M5_chains.png, width=1800, height=1350, dpi=300, ray=1
color plddt_vl, M5
color plddt_l, M5 and b > 50
color plddt_h, M5 and b > 70
color plddt_vh, M5 and b > 90
png figs/M5_plddt.png, width=1800, height=1350, dpi=300, ray=1
disable all
enable C10
util.cbc C10
color 0xD55E00, C10 and chain A
show spheres, C10 and chain A and not hydro
set sphere_scale, 0.45, C10 and chain A
set cartoon_transparency, 0.45, C10 and not chain A
png figs/C10_chains.png, width=1800, height=1350, dpi=300, ray=1
color plddt_vl, C10
color plddt_l, C10 and b > 50
color plddt_h, C10 and b > 70
color plddt_vh, C10 and b > 90
png figs/C10_plddt.png, width=1800, height=1350, dpi=300, ray=1
disable all
enable C6
util.cbc C6
color 0xD55E00, C6 and chain A
show spheres, C6 and chain A and not hydro
set sphere_scale, 0.45, C6 and chain A
set cartoon_transparency, 0.45, C6 and not chain A
png figs/C6_chains.png, width=1800, height=1350, dpi=300, ray=1
color plddt_vl, C6
color plddt_l, C6 and b > 50
color plddt_h, C6 and b > 70
color plddt_vh, C6 and b > 90
png figs/C6_plddt.png, width=1800, height=1350, dpi=300, ray=1
disable all
enable C2
util.cbc C2
color 0xD55E00, C2 and chain A
show spheres, C2 and chain A and not hydro
set sphere_scale, 0.45, C2 and chain A
set cartoon_transparency, 0.45, C2 and not chain A
png figs/C2_chains.png, width=1800, height=1350, dpi=300, ray=1
color plddt_vl, C2
color plddt_l, C2 and b > 50
color plddt_h, C2 and b > 70
color plddt_vh, C2 and b > 90
png figs/C2_plddt.png, width=1800, height=1350, dpi=300, ray=1
disable all
enable C7
util.cbc C7
color 0xD55E00, C7 and chain A
show spheres, C7 and chain A and not hydro
set sphere_scale, 0.45, C7 and chain A
set cartoon_transparency, 0.45, C7 and not chain A
png figs/C7_chains.png, width=1800, height=1350, dpi=300, ray=1
color plddt_vl, C7
color plddt_l, C7 and b > 50
color plddt_h, C7 and b > 70
color plddt_vh, C7 and b > 90
png figs/C7_plddt.png, width=1800, height=1350, dpi=300, ray=1
disable all
enable C1
util.cbc C1
color 0xD55E00, C1 and chain A
show spheres, C1 and chain A and not hydro
set sphere_scale, 0.45, C1 and chain A
set cartoon_transparency, 0.45, C1 and not chain A
png figs/C1_chains.png, width=1800, height=1350, dpi=300, ray=1
color plddt_vl, C1
color plddt_l, C1 and b > 50
color plddt_h, C1 and b > 70
color plddt_vh, C1 and b > 90
png figs/C1_plddt.png, width=1800, height=1350, dpi=300, ray=1
disable all
enable C4
util.cbc C4
color 0xD55E00, C4 and chain A
show spheres, C4 and chain A and not hydro
set sphere_scale, 0.45, C4 and chain A
set cartoon_transparency, 0.45, C4 and not chain A
png figs/C4_chains.png, width=1800, height=1350, dpi=300, ray=1
color plddt_vl, C4
color plddt_l, C4 and b > 50
color plddt_h, C4 and b > 70
color plddt_vh, C4 and b > 90
png figs/C4_plddt.png, width=1800, height=1350, dpi=300, ray=1
disable all
enable C0
util.cbc C0
color 0xD55E00, C0 and chain A
show spheres, C0 and chain A and not hydro
set sphere_scale, 0.45, C0 and chain A
set cartoon_transparency, 0.45, C0 and not chain A
png figs/C0_chains.png, width=1800, height=1350, dpi=300, ray=1
color plddt_vl, C0
color plddt_l, C0 and b > 50
color plddt_h, C0 and b > 70
color plddt_vh, C0 and b > 90
png figs/C0_plddt.png, width=1800, height=1350, dpi=300, ray=1
disable all
enable C9
util.cbc C9
color 0xD55E00, C9 and chain A
show spheres, C9 and chain A and not hydro
set sphere_scale, 0.45, C9 and chain A
set cartoon_transparency, 0.45, C9 and not chain A
png figs/C9_chains.png, width=1800, height=1350, dpi=300, ray=1
color plddt_vl, C9
color plddt_l, C9 and b > 50
color plddt_h, C9 and b > 70
color plddt_vh, C9 and b > 90
png figs/C9_plddt.png, width=1800, height=1350, dpi=300, ray=1
disable all
enable M1
color 0x0072B2, M1
hide spheres, M1
set cartoon_transparency, 0, (M1)
enable M2
color 0xD55E00, M2
hide spheres, M2
set cartoon_transparency, 0, (M2)
enable M3
color 0x009E73, M3
hide spheres, M3
set cartoon_transparency, 0, (M3)
enable M4
color 0xCC79A7, M4
hide spheres, M4
set cartoon_transparency, 0, (M4)
enable M5
color 0xE69F00, M5
hide spheres, M5
set cartoon_transparency, 0, (M5)
enable C10
color 0x0072B2, C10
hide spheres, C10
set cartoon_transparency, 0, (C10)
enable C6
color 0xD55E00, C6
hide spheres, C6
set cartoon_transparency, 0, (C6)
enable C2
color 0x009E73, C2
hide spheres, C2
set cartoon_transparency, 0, (C2)
enable C7
color 0xCC79A7, C7
hide spheres, C7
set cartoon_transparency, 0, (C7)
enable C1
color 0xE69F00, C1
hide spheres, C1
set cartoon_transparency, 0, (C1)
enable C4
color 0x0072B2, C4
hide spheres, C4
set cartoon_transparency, 0, (C4)
enable C0
color 0xD55E00, C0
hide spheres, C0
set cartoon_transparency, 0, (C0)
enable C9
color 0x009E73, C9
hide spheres, C9
set cartoon_transparency, 0, (C9)
png figs/overlay.png, width=2400, height=1700, dpi=300, ray=1
load <WORK>/targets/H1388/report/templates/6BYJ_F.cif, tpl_6BYJ_F
disable all
enable tpl_6BYJ_F
hide everything, tpl_6BYJ_F
show cartoon, tpl_6BYJ_F
color 0xBBBBBB, tpl_6BYJ_F
color 0x5B8DB8, tpl_6BYJ_F and chain F
show sticks, tpl_6BYJ_F and organic
orient tpl_6BYJ_F
zoom tpl_6BYJ_F, 0
png figs/tpl_6BYJ_F.png, width=1800, height=1350, dpi=300, ray=1
load <WORK>/targets/H1388/report/templates/6S9K_A.cif, tpl_6S9K_A
disable all
enable tpl_6S9K_A
hide everything, tpl_6S9K_A
show cartoon, tpl_6S9K_A
color 0xBBBBBB, tpl_6S9K_A
color 0x5B8DB8, tpl_6S9K_A and chain A
show sticks, tpl_6S9K_A and organic
color 0xD55E00, tpl_6S9K_A and chain B
show sticks, tpl_6S9K_A and chain B and not name N+C+O
face tpl_6S9K_A and chain B, tpl_6S9K_A and chain A, tpl_6S9K_A, 0
png figs/tpl_6S9K_A.png, width=1800, height=1350, dpi=300, ray=1
load <WORK>/targets/H1388/report/templates/6Y6B_B.cif, tpl_6Y6B_B
disable all
enable tpl_6Y6B_B
hide everything, tpl_6Y6B_B
show cartoon, tpl_6Y6B_B
color 0xBBBBBB, tpl_6Y6B_B
color 0x5B8DB8, tpl_6Y6B_B and chain B
show sticks, tpl_6Y6B_B and organic
color 0xD55E00, tpl_6Y6B_B and chain D
show sticks, tpl_6Y6B_B and chain D and not name N+C+O
face tpl_6Y6B_B and chain D, tpl_6Y6B_B and chain B, tpl_6Y6B_B, 0
png figs/tpl_6Y6B_B.png, width=1800, height=1350, dpi=300, ray=1
load <WORK>/targets/H1388/report/templates/7V9B_A.cif, tpl_7V9B_A
disable all
enable tpl_7V9B_A
hide everything, tpl_7V9B_A
show cartoon, tpl_7V9B_A
color 0xBBBBBB, tpl_7V9B_A
color 0x5B8DB8, tpl_7V9B_A and chain A
show sticks, tpl_7V9B_A and organic
color 0xD55E00, tpl_7V9B_A and chain B
show sticks, tpl_7V9B_A and chain B and not name N+C+O
face tpl_7V9B_A and chain B, tpl_7V9B_A and chain A, tpl_7V9B_A, 0
png figs/tpl_7V9B_A.png, width=1800, height=1350, dpi=300, ray=1
load <WORK>/targets/H1388/report/templates/6BZD_B.cif, tpl_6BZD_B
disable all
enable tpl_6BZD_B
hide everything, tpl_6BZD_B
show cartoon, tpl_6BZD_B
color 0xBBBBBB, tpl_6BZD_B
color 0x5B8DB8, tpl_6BZD_B and chain B
show sticks, tpl_6BZD_B and organic
color 0xD55E00, tpl_6BZD_B and chain K
show sticks, tpl_6BZD_B and chain K and not name N+C+O
face tpl_6BZD_B and chain K, tpl_6BZD_B and chain B, tpl_6BZD_B, 0
png figs/tpl_6BZD_B.png, width=1800, height=1350, dpi=300, ray=1
load <WORK>/targets/H1388/report/templates/6ZC9_A.cif, tpl_6ZC9_A
disable all
enable tpl_6ZC9_A
hide everything, tpl_6ZC9_A
show cartoon, tpl_6ZC9_A
color 0xBBBBBB, tpl_6ZC9_A
color 0x5B8DB8, tpl_6ZC9_A and chain A
show sticks, tpl_6ZC9_A and organic
color 0xD55E00, tpl_6ZC9_A and chain E
show sticks, tpl_6ZC9_A and chain E and not name N+C+O
face tpl_6ZC9_A and chain E, tpl_6ZC9_A and chain A, tpl_6ZC9_A, 0
png figs/tpl_6ZC9_A.png, width=1800, height=1350, dpi=300, ray=1
