set lod_medium 400000
set lod_low 400000
set bg white
set sm relative
set csd 24
set outline on
load <WORK>/targets/H1388/report/figs/pep_split.pdb
show cartoon chain A+P+B+U+O
color #C8C8C8 chain R
show sticks chain R and resi 49+56+131+132
color #0072B2 chain R and resi 49+56+131+132
show sticks chain A+P+B+U+O
color #D55E00 chain A
color #009E73 chain P
color #CC79A7 chain B
color #E69F00 chain U
color #56B4E9 chain O
orient chain A+P+B+U+O
screenshot figs/pep_split.png 1800 1350 300
