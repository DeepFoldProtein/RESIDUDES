import csv, sys, gemmi, numpy as np
rows=[r for r in csv.DictReader(open(sys.argv[1])) if r['pass']=='1']
def ca(p):
    m=gemmi.read_structure(p)[0]; return {c.name:{r.seqid.num:np.array(r['CA'][0].pos.tolist()) for r in c if r.find_atom('CA','*')} for c in m}
ref=ca('<WORK>/targets/H1388/results/A1B1/sep/af3/seed_20/sample_0/model.cif')
out=[]
for r in rows:
    c=ca(r['path']); ks=[k for k in ref['B'] if k in c['B']]
    P=np.array([c['B'][k] for k in ks]); Q=np.array([ref['B'][k] for k in ks])
    pc,qc=P.mean(0),Q.mean(0); U,S,Vt=np.linalg.svd((P-pc).T@(Q-qc)); d=np.sign(np.linalg.det(Vt.T@U.T)); R=Vt.T@np.diag([1,1,d])@U.T
    pk=[k for k in range(4,11)]; X=(np.array([c['A'][k] for k in pk])-pc)@R.T+qc; Y=np.array([ref['A'][k] for k in pk])
    out.append((r['runner'],float(r['ranking_score']),r['key'],np.sqrt(((X-Y)**2).sum(1).mean())))
for run in ['af3','pt2','bt2','of3']:
    v=sorted([o for o in out if o[0]==run],key=lambda o:-o[1])
    print(run,'pass',len(v),'core<2A',sum(o[3]<2 for o in v),'top:',' '.join('%s(%.3f,%.1fA)'%(o[2].split('__')[-1],o[1],o[3]) for o in v[:6]))
    if run in('bt2','of3'): print('  best consensus:',' '.join('%s(%.3f,%.1fA)'%(o[2].split('__')[-1],o[1],o[3]) for o in [o for o in v if o[3]<2][:3]))
