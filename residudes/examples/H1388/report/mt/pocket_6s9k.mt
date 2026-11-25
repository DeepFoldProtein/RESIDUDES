set lod_medium 400000
set lod_low 400000
set bg white
set sm relative
set csd 24
set outline on
load <WORK>/targets/H1388/dl/pdb/6S9K.pdb
show cartoon
color chain
color #A0A0A0 chain A
show spacefill chain B
color #D55E00 chain B
show sticks chain A and resi 50+57+132+133
color #0072B2 chain A and resi 50+57+132+133
orient
screenshot figs/pocket_6s9k.png 1800 1350 300
