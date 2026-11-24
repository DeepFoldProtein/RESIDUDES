# lantern scenes: written by `lantern slides`; run with `pymol -cq scenes.pml` from report/
reinitialize
bg_color white
set ray_opaque_background, 1
set antialias, 2
set ray_shadows, 0
set cartoon_fancy_helices, 1
set ray_trace_mode, 1
set ray_trace_color, 0x404040
set_color plddt_vh, [0.000, 0.325, 0.839]
set_color plddt_h, [0.396, 0.796, 0.953]
set_color plddt_l, [1.000, 0.859, 0.075]
set_color plddt_vl, [1.000, 0.490, 0.271]
load /gpfs/deepfold/casp/casp17/targets/H1388/results/A1B1/af3_msa_pS548/af3/seed_1/sample_0/model.cif, A1B1
load /gpfs/deepfold/casp/casp17/targets/H1388/results/A2B2/af3_msa_pS548/af3/seed_1/sample_0/model.cif, A2B2
load /gpfs/deepfold/casp/casp17/targets/H1388/results/A1B1/mmseqs2_gpu/bt2_jax/seed_8/sample_1/model.cif, C0
load /gpfs/deepfold/casp/casp17/targets/H1388/results/A1B1/af3_msa_pS548/af3/seed_19/sample_3/model.cif, C1
load /gpfs/deepfold/casp/casp17/targets/H1388/results/A1B1/af3_msa/af3/seed_10/sample_0/model.cif, C8
load /gpfs/deepfold/casp/casp17/targets/H1388/results/A1B1/af3_msa/bt2_jax/seed_18/sample_4/model.cif, C7
load /gpfs/deepfold/casp/casp17/targets/H1388/results/A1B1/mmseqs2_gpu/af3/seed_15/sample_3/model.cif, C5
load /gpfs/deepfold/casp/casp17/targets/H1388/results/A1B1/af3_msa/of3/seed_14/sample_4/model.cif, C3
load /gpfs/deepfold/casp/casp17/targets/H1388/results/A1B1/mmseqs2_gpu/af3/seed_17/sample_4/model.cif, C2
load /gpfs/deepfold/casp/casp17/targets/H1388/results/A1B1/af3_msa/of3/seed_16/sample_0/model.cif, C6
remove solvent
remove hydro
hide everything
show cartoon
super A2B2 and name CA, A1B1 and name CA
super C0 and name CA, A1B1 and name CA
super C1 and name CA, A1B1 and name CA
super C8 and name CA, A1B1 and name CA
super C7 and name CA, A1B1 and name CA
super C5 and name CA, A1B1 and name CA
super C3 and name CA, A1B1 and name CA
super C2 and name CA, A1B1 and name CA
super C6 and name CA, A1B1 and name CA
orient A1B1
zoom A1B1, 4
disable all
enable A1B1
util.cbc A1B1
color 0xD55E00, A1B1 and chain A
show spheres, A1B1 and chain A and not hydro
set sphere_scale, 0.45, A1B1 and chain A
set cartoon_transparency, 0.45, A1B1 and not chain A
png figs/A1B1_chains.png, width=1000, height=750, dpi=120, ray=1
color plddt_vl, A1B1
color plddt_l, A1B1 and b > 50
color plddt_h, A1B1 and b > 70
color plddt_vh, A1B1 and b > 90
png figs/A1B1_plddt.png, width=1000, height=750, dpi=120, ray=1
disable all
enable A2B2
util.cbc A2B2
color 0xD55E00, A2B2 and chain A
show spheres, A2B2 and chain A and not hydro
set sphere_scale, 0.45, A2B2 and chain A
set cartoon_transparency, 0.45, A2B2 and not chain A
png figs/A2B2_chains.png, width=1000, height=750, dpi=120, ray=1
color plddt_vl, A2B2
color plddt_l, A2B2 and b > 50
color plddt_h, A2B2 and b > 70
color plddt_vh, A2B2 and b > 90
png figs/A2B2_plddt.png, width=1000, height=750, dpi=120, ray=1
disable all
enable C0
util.cbc C0
color 0xD55E00, C0 and chain A
show spheres, C0 and chain A and not hydro
set sphere_scale, 0.45, C0 and chain A
set cartoon_transparency, 0.45, C0 and not chain A
png figs/C0_chains.png, width=1000, height=750, dpi=120, ray=1
color plddt_vl, C0
color plddt_l, C0 and b > 50
color plddt_h, C0 and b > 70
color plddt_vh, C0 and b > 90
png figs/C0_plddt.png, width=1000, height=750, dpi=120, ray=1
disable all
enable C1
util.cbc C1
color 0xD55E00, C1 and chain A
show spheres, C1 and chain A and not hydro
set sphere_scale, 0.45, C1 and chain A
set cartoon_transparency, 0.45, C1 and not chain A
png figs/C1_chains.png, width=1000, height=750, dpi=120, ray=1
color plddt_vl, C1
color plddt_l, C1 and b > 50
color plddt_h, C1 and b > 70
color plddt_vh, C1 and b > 90
png figs/C1_plddt.png, width=1000, height=750, dpi=120, ray=1
disable all
enable C8
util.cbc C8
color 0xD55E00, C8 and chain A
show spheres, C8 and chain A and not hydro
set sphere_scale, 0.45, C8 and chain A
set cartoon_transparency, 0.45, C8 and not chain A
png figs/C8_chains.png, width=1000, height=750, dpi=120, ray=1
color plddt_vl, C8
color plddt_l, C8 and b > 50
color plddt_h, C8 and b > 70
color plddt_vh, C8 and b > 90
png figs/C8_plddt.png, width=1000, height=750, dpi=120, ray=1
disable all
enable C7
util.cbc C7
color 0xD55E00, C7 and chain A
show spheres, C7 and chain A and not hydro
set sphere_scale, 0.45, C7 and chain A
set cartoon_transparency, 0.45, C7 and not chain A
png figs/C7_chains.png, width=1000, height=750, dpi=120, ray=1
color plddt_vl, C7
color plddt_l, C7 and b > 50
color plddt_h, C7 and b > 70
color plddt_vh, C7 and b > 90
png figs/C7_plddt.png, width=1000, height=750, dpi=120, ray=1
disable all
enable C5
util.cbc C5
color 0xD55E00, C5 and chain A
show spheres, C5 and chain A and not hydro
set sphere_scale, 0.45, C5 and chain A
set cartoon_transparency, 0.45, C5 and not chain A
png figs/C5_chains.png, width=1000, height=750, dpi=120, ray=1
color plddt_vl, C5
color plddt_l, C5 and b > 50
color plddt_h, C5 and b > 70
color plddt_vh, C5 and b > 90
png figs/C5_plddt.png, width=1000, height=750, dpi=120, ray=1
disable all
enable C3
util.cbc C3
color 0xD55E00, C3 and chain A
show spheres, C3 and chain A and not hydro
set sphere_scale, 0.45, C3 and chain A
set cartoon_transparency, 0.45, C3 and not chain A
png figs/C3_chains.png, width=1000, height=750, dpi=120, ray=1
color plddt_vl, C3
color plddt_l, C3 and b > 50
color plddt_h, C3 and b > 70
color plddt_vh, C3 and b > 90
png figs/C3_plddt.png, width=1000, height=750, dpi=120, ray=1
disable all
enable C2
util.cbc C2
color 0xD55E00, C2 and chain A
show spheres, C2 and chain A and not hydro
set sphere_scale, 0.45, C2 and chain A
set cartoon_transparency, 0.45, C2 and not chain A
png figs/C2_chains.png, width=1000, height=750, dpi=120, ray=1
color plddt_vl, C2
color plddt_l, C2 and b > 50
color plddt_h, C2 and b > 70
color plddt_vh, C2 and b > 90
png figs/C2_plddt.png, width=1000, height=750, dpi=120, ray=1
disable all
enable C6
util.cbc C6
color 0xD55E00, C6 and chain A
show spheres, C6 and chain A and not hydro
set sphere_scale, 0.45, C6 and chain A
set cartoon_transparency, 0.45, C6 and not chain A
png figs/C6_chains.png, width=1000, height=750, dpi=120, ray=1
color plddt_vl, C6
color plddt_l, C6 and b > 50
color plddt_h, C6 and b > 70
color plddt_vh, C6 and b > 90
png figs/C6_plddt.png, width=1000, height=750, dpi=120, ray=1
disable all
enable A1B1
color 0x0072B2, A1B1
enable C0
color 0xD55E00, C0
enable C1
color 0x009E73, C1
enable C8
color 0xCC79A7, C8
enable C7
color 0xE69F00, C7
enable C5
color 0x0072B2, C5
enable C3
color 0xD55E00, C3
enable C2
color 0x009E73, C2
enable C6
color 0xCC79A7, C6
orient A1B1
png figs/overlay.png, width=1200, height=850, dpi=120, ray=1
