import csv, sys, gemmi, numpy as np
from collections import defaultdict
rows=list(csv.DictReader(open(sys.argv[1])))
def ca(path):
    st=gemmi.read_structure(path); m=st[0]
    out={}
    for ch in m:
        out[ch.name]={r.seqid.num:np.array(r['CA'][0].pos.tolist()) for r in ch if r.find_atom('CA','*')}
    return out
def kabsch(P,Q):
    pc,qc=P.mean(0),Q.mean(0); H=(P-pc).T@(Q-qc); U,S,Vt=np.linalg.svd(H)
    d=np.sign(np.linalg.det(Vt.T@U.T)); D=np.diag([1,1,d]); R=Vt.T@D@U.T
    return R,pc,qc
best={}
for r in rows:
    k=(r['variant'],r['runner'])
    if k not in best or float(r['ranking_score'])>float(best[k]['ranking_score']): best[k]=r
ref=ca(best[('sep','af3')]['path'])
chs=sorted(ref); rec,pep=chs[1],chs[0]
keys=sorted(set(ref[rec]))
stats=defaultdict(list)
for r in rows:
    c=ca(r['path'])
    ks=[k for k in keys if k in c[rec]]
    P=np.array([c[rec][k] for k in ks]); Q=np.array([ref[rec][k] for k in ks])
    R,pc,qc=kabsch(P,Q)
    pk=[k for k in ref[pep] if k in c[pep]]
    X=(np.array([c[pep][k] for k in pk])-pc)@R.T+qc; Y=np.array([ref[pep][k] for k in pk])
    core=[i for i,k in enumerate(pk) if 4<=k<=10]
    rm=np.sqrt(((X-Y)**2).sum(1).mean()); rc=np.sqrt(((X[core]-Y[core])**2).sum(1).mean())
    stats[(r['variant'],r['runner'])].append((rm,rc,float(r['ranking_score']),r['seed'],r['sample']))
for k,v in sorted(stats.items()):
    a=np.array([x[:2] for x in v]); top=max(v,key=lambda x:x[2])
    print(k,'n',len(v),'pepRMSD med %.2f core(4-10) med %.2f; frac core<2A %.2f; top-rs model pep %.2f core %.2f (s%s_%s)'%(np.median(a[:,0]),np.median(a[:,1]),(a[:,1]<2).mean(),top[0],top[1],top[3],top[4]))
