load /store/deepfold/lantern/targets/H1388/dl/pdb/6A5S.cif, t
remove solvent
remove not polymer
select gam, chain A+B
select pep, (chain C+E+F+H) and byres (all within 5 of gam)
remove not (gam or pep)
hide everything
bg_color white
show cartoon, gam
color grey80, chain A
color grey60, chain B
show sticks, pep and not hydro
color orange, pep and elem C
select pocket, gam and resi 50+57+132+133
show sticks, pocket and not hydro
color marine, pocket and elem C
color red, resn SEP and elem C
set label_size, 18
label chain A and pocket and name CA, "%s%s" % (resn, resi)
orient chain A
set ray_opaque_background, 1
png figs/template_6a5s.png, width=1600, height=1200, dpi=150, ray=1
