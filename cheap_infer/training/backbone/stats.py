"""Fit every normalization constant on a fixed training-only sample."""
import json,hashlib
import numpy as np
from common import *
from data import groups
class Moments:
    def __init__(self):self.n=0;self.s=None;self.ss=None
    def add(self,x):
        x=np.asarray(x,dtype=np.float64).reshape(-1,x.shape[-1]);assert np.isfinite(x).all()
        a=x.sum(0);aa=np.einsum('ij,ij->j',x,x)
        self.s=a if self.s is None else self.s+a;self.ss=aa if self.ss is None else self.ss+aa;self.n+=len(x)
    def result(self):
        mean=self.s/self.n;std=np.sqrt(np.maximum(self.ss/self.n-mean**2,0));std=np.maximum(std,np.maximum(abs(mean)*.01,1e-3))
        return dict(n=self.n,mean=mean.tolist(),std=std.tolist())
def main():
    rows=[r for r in json.loads((ROOT/'manifest.json').read_text()) if r['split']=='train']
    rows=sorted(rows,key=lambda r:hashlib.sha256(('stats/'+str(r['i'])).encode()).hexdigest())[:8192]
    cheap,aim,teacher=[Reader(x) for x in ('cheap','aim','teacher')]
    xtx=np.zeros((119,119));xty=np.zeros(119)
    for row in rows:
        r=cheap.get(row['i']);a=aim.get(row['i'])[0]
        if a['energy_eV'] is None:continue
        counts=np.bincount(r['top']['z'],minlength=119).astype(float);xtx+=np.outer(counts,counts);xty+=counts*a['energy_eV']
    coeff=np.linalg.lstsq(xtx+np.eye(119)*1e-6,xty,rcond=1e-12)[0]
    fm={};tm={};nm=Moments();em=Moments()
    for row in rows:
        r=cheap.get(row['i']);aa=aim.get(row['i']);tt=teacher.get(row['i'])['targets'];nm.add(r['node'][:,:18]);em.add(r['edge'][:,:19])
        for v,a,t in zip(r['views'],aa,tt):
            gg=groups(r,v,a);gg['aim_global']=gg['aim_global'].copy();gg['aim_global'][0,1]-=coeff[r['top']['z']].mean() if a['energy_eV'] is not None else 0
            for k,x in gg.items():fm.setdefault(k,Moments()).add(x.astype(np.float32))
            for k,x in t.items():tm.setdefault(k,Moments()).add(x.astype(np.float32))
    result=dict(features={k:v.result() for k,v in fm.items()},teacher={k:v.result() for k,v in tm.items()},node_qc=nm.result(),edge_qc=em.result(),energy_coeff=coeff.tolist(),training_ids=[r['i'] for r in rows],description='train-only deterministic max8192 records; all available views; elemental AIM energy baseline, not isolated-atom cohesive energy')
    atomic_json('stats.json',result)
if __name__=='__main__':main()
