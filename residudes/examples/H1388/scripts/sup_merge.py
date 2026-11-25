import gemmi, numpy as np, sys
R='<WORK>/targets/H1388/results/A1B1/'
ms={'A':R+'sep/af3/seed_20/sample_0/model.cif','P':R+'sep/pt2/seed_5/sample_0/model.cif','B':R+'sep/bt2/seed_1/sample_0/model.cif','U':R+'base/af3/seed_5/sample_0/model.cif','O':R+'sep/of3/seed_6/sample_0/model.cif'}
def load(p):
    st=gemmi.read_structure(p); st.setup_entities(); return st
ref=load(ms['A']); rm=ref[0]
recname=[c.name for c in rm][1]; pepname=[c.name for c in rm][0]
refca={r.seqid.num:r['CA'][0].pos for r in rm[recname]}
out=gemmi.Structure(); out.add_model(gemmi.Model('1'))
rc=rm[recname].clone(); rc.name='R'; out[0].add_chain(rc)
for k,p in ms.items():
    st=load(p); m=st[0]; ch=[c.name for c in m]
    rec=m[ch[1]]; pep=m[ch[0]]
    a=[];b=[]
    for r in rec:
        if r.seqid.num in refca and r.find_atom('CA','*'): a.append(r['CA'][0].pos); b.append(refca[r.seqid.num])
    sup=gemmi.superpose_positions(b,a)
    pc=pep.clone(); pc.name=k
    for r in pc:
        for at in r: at.pos=gemmi.Position(*sup.transform.apply(at.pos).tolist()) if hasattr(sup.transform.apply(at.pos),'tolist') else sup.transform.apply(at.pos)
    out[0].add_chain(pc)
out.setup_entities()
out.write_pdb(sys.argv[1]); print('ok',recname,pepname)
