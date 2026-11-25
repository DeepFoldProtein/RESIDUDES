# residudes scenes: written by `residudes slides`; run with `molterm --no-tui --script scenes.mt` from report/
set lod_medium 400000
set lod_low 400000
set bg white
set sm relative
set csd 24
set ch 1.6
set csh 1.8
set cl 0.30
set outline on
set ot 0.2
set od 0.2
set fog 0.4
load <WORK>/targets/H1388/results/A1B1/sep/af3/seed_20/sample_0/model.cif
rename model M1
load <WORK>/targets/H1388/results/A1B1/sep/pt2/seed_5/sample_0/model.cif
rename model M2
load <WORK>/targets/H1388/results/A1B1/sep/af3/seed_9/sample_0/model.cif
rename model M3
load <WORK>/targets/H1388/results/A1B1/sep/pt2/seed_10/sample_0/model.cif
rename model M4
load <WORK>/targets/H1388/results/A1B1/sep/af3/seed_7/sample_0/model.cif
rename model M5
load <WORK>/targets/H1388/results/A1B1/sep/af3/seed_14/sample_2/model.cif
rename model C10
load <WORK>/targets/H1388/results/A1B1/sep/pt2/seed_4/sample_0/model.cif
rename model C6
load <WORK>/targets/H1388/results/A1B1/sep/af3/seed_15/sample_3/model.cif
rename model C2
load <WORK>/targets/H1388/results/A1B1/sep/of3/seed_6/sample_4/model.cif
rename model C7
load <WORK>/targets/H1388/results/A1B1/base/af3/seed_2/sample_2/model.cif
rename model C1
load <WORK>/targets/H1388/results/A1B1/base/af3/seed_5/sample_3/model.cif
rename model C4
load <WORK>/targets/H1388/results/A1B1/sep/of3/seed_1/sample_2/model.cif
rename model C0
load <WORK>/targets/H1388/results/A1B1/sep/pt2/seed_2/sample_0/model.cif
rename model C9
align M2 to M1 automap
align M3 to M1 automap
align M4 to M1 automap
align M5 to M1 automap
align C10 to M1 automap
align C6 to M1 automap
align C2 to M1 automap
align C7 to M1 automap
align C1 to M1 automap
align C4 to M1 automap
align C0 to M1 automap
align C9 to M1 automap
show cartoon
orient M1
disable all
enable M1
color chain
show spacefill chain A
color #D55E00 chain A
screenshot figs/M1_chains.png 1800 1350 300
color plddt
screenshot figs/M1_plddt.png 1800 1350 300
disable all
enable M2
color chain
show spacefill chain A
color #D55E00 chain A
screenshot figs/M2_chains.png 1800 1350 300
color plddt
screenshot figs/M2_plddt.png 1800 1350 300
disable all
enable M3
color chain
show spacefill chain A
color #D55E00 chain A
screenshot figs/M3_chains.png 1800 1350 300
color plddt
screenshot figs/M3_plddt.png 1800 1350 300
disable all
enable M4
color chain
show spacefill chain A
color #D55E00 chain A
screenshot figs/M4_chains.png 1800 1350 300
color plddt
screenshot figs/M4_plddt.png 1800 1350 300
disable all
enable M5
color chain
show spacefill chain A
color #D55E00 chain A
screenshot figs/M5_chains.png 1800 1350 300
color plddt
screenshot figs/M5_plddt.png 1800 1350 300
disable all
enable C10
color chain
show spacefill chain A
color #D55E00 chain A
screenshot figs/C10_chains.png 1800 1350 300
color plddt
screenshot figs/C10_plddt.png 1800 1350 300
disable all
enable C6
color chain
show spacefill chain A
color #D55E00 chain A
screenshot figs/C6_chains.png 1800 1350 300
color plddt
screenshot figs/C6_plddt.png 1800 1350 300
disable all
enable C2
color chain
show spacefill chain A
color #D55E00 chain A
screenshot figs/C2_chains.png 1800 1350 300
color plddt
screenshot figs/C2_plddt.png 1800 1350 300
disable all
enable C7
color chain
show spacefill chain A
color #D55E00 chain A
screenshot figs/C7_chains.png 1800 1350 300
color plddt
screenshot figs/C7_plddt.png 1800 1350 300
disable all
enable C1
color chain
show spacefill chain A
color #D55E00 chain A
screenshot figs/C1_chains.png 1800 1350 300
color plddt
screenshot figs/C1_plddt.png 1800 1350 300
disable all
enable C4
color chain
show spacefill chain A
color #D55E00 chain A
screenshot figs/C4_chains.png 1800 1350 300
color plddt
screenshot figs/C4_plddt.png 1800 1350 300
disable all
enable C0
color chain
show spacefill chain A
color #D55E00 chain A
screenshot figs/C0_chains.png 1800 1350 300
color plddt
screenshot figs/C0_plddt.png 1800 1350 300
disable all
enable C9
color chain
show spacefill chain A
color #D55E00 chain A
screenshot figs/C9_chains.png 1800 1350 300
color plddt
screenshot figs/C9_plddt.png 1800 1350 300
disable all
enable M1
color #0072B2 M1
enable M2
color #D55E00 M2
enable M3
color #009E73 M3
enable M4
color #CC79A7 M4
enable M5
color #E69F00 M5
enable C10
color #0072B2 C10
enable C6
color #D55E00 C6
enable C2
color #009E73 C2
enable C7
color #CC79A7 C7
enable C1
color #E69F00 C1
enable C4
color #0072B2 C4
enable C0
color #D55E00 C0
enable C9
color #009E73 C9
orient M1
screenshot figs/overlay.png 2400 1700 300
