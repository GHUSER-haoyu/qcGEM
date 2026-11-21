"""Read-only mmap source, atom-order checks, offline cheap features, grouped split."""
import argparse, concurrent.futures, json, multiprocessing as mp, os, time, traceback
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from common import *
from features import reconstruct,topology,build_views,identity,nonstereo_identity
D=S=None
def init():
    global D,S
    torch.set_num_threads(1)
    D,S=torch.load(SOURCE,weights_only=False,mmap=True,map_location='cpu')
def raw(i):
    def get(k):
        lo,hi=map(int,S[k][i:i+2]);return getattr(D,k)[lo:hi].numpy().copy()
    n=get('node_features');e=get('edge_features');xyz=get('xyz');nn=len(n)
    lo,hi=map(int,S['edge_index'][i:i+2]);idx=D.edge_index[:,lo:hi].numpy()
    expected=np.array([np.repeat(np.arange(nn),nn),np.tile(np.arange(nn),nn)])
    assert np.array_equal(idx,expected),'source edge order differs'
    assert n.shape==(nn,80) and e.shape==(nn*nn,53) and xyz.shape==(nn,3)
    assert all(np.isfinite(a).all() for a in (n,e,xyz))
    return dict(i=i,cid=D.CID[i],smi=D.smi[i],node=n,edge=e,xyz=xyz,md=get('global_MD').reshape(-1),fp=get('global_FP').reshape(-1))
def shard(job):
    si,limit=job;report=ROOT/f'reports/prepare/{si:04d}.json'
    if report.exists():return json.loads(report.read_text())
    db=open_db('cheap',si,False);rows=[];errors=[];start=time.time()
    for i in range(si*SHARD_SIZE,min((si+1)*SHARD_SIZE,limit)):
        event(f'logs/prepare_{si:04d}.jsonl',stage='begin',i=i)
        try:
            with db.begin() as txn:prior=txn.get(key(i))
            if prior is not None:
                r=pickle.loads(prior)
                rows.append(dict(i=i,cid=r['cid'],canonical=r['canonical'],family=r['family'],scaffold=r['scaffold'],n=len(r['xyz']),views=len(r['views']),charge=int(r['node'][:,20].sum())))
                event(f'logs/prepare_{si:04d}.jsonl',stage='reuse_committed',i=i)
                continue
            r=raw(i);m=reconstruct(r['node'][:,18:],r['edge'][:,19:],r['xyz'])
            canon,family,scaffold=identity(m)
            # Source SMILES is an independent identity check only, never atom ordering.
            original=Chem.MolFromSmiles(r['smi'])
            if original is None:raise ValueError('source SMILES invalid')
            if nonstereo_identity(original)!=family:raise ValueError('source topology disagrees with graph')
            top=topology(m,r['node'][:,18:],r['edge'][:,19:]);views,states=build_views(m,top)
            r.update(top=top,views=views,view_states=states,molblock=Chem.MolToMolBlock(m),canonical=canon,family=family,scaffold=scaffold)
            # Continuous cheap caches use fp16; QC labels and xyz stay fp32.
            for v in views:
                for k,a in v.items():
                    if isinstance(a,np.ndarray) and k!='xyz':
                        if np.max(np.abs(a),initial=0)<65000:v[k]=a.astype(np.float16)
            put(db,i,r)
            rows.append(dict(i=i,cid=r['cid'],canonical=canon,family=family,scaffold=scaffold,n=len(r['xyz']),views=len(views),charge=int(Chem.GetFormalCharge(m))))
            event(f'logs/prepare_{si:04d}.jsonl',stage='done',i=i,view_states=states,feature_seconds=[v['feature_seconds'] for v in views])
        except Exception as ex:
            source_valid=Chem.MolFromSmiles(D.smi[i]) is not None
            errors.append(dict(i=i,cid=D.CID[i],source_smiles_valid=source_valid,error=type(ex).__name__+': '+str(ex),traceback=traceback.format_exc(limit=3)))
        if (i+1)%32==0:event(f'logs/prepare_{si:04d}.jsonl',last=i,success=len(rows),failed=len(errors),elapsed=time.time()-start)
    db.sync();db.close()
    result=dict(shard=si,rows=rows,errors=errors,elapsed=time.time()-start)
    atomic_json(f'reports/prepare/{si:04d}.json',result);return result
def external_families():
    families=set();provenance=[]
    for p in [Path('/run/ti/qcgem_official_downstream_recheck_20261014/rows.json'),Path('/run/ti/qcgem_official_advantage_20261014/rows.json')]:
        rows=json.loads(p.read_text());provenance.append(dict(path=str(p),sha256=sha(p),rows=len(rows)))
        for row in rows:
            smi=row.get('canonical_isomeric_smiles',row.get('identity',row.get('smiles')))
            if smi is None:raise ValueError('missing external identity '+str(row.keys()))
            m=Chem.MolFromSmiles(smi)
            if m is not None:families.add(nonstereo_identity(m))
    return families,provenance
def finish(limit):
    rows=[];errors=[]
    for si in range((limit+SHARD_SIZE-1)//SHARD_SIZE):
        d=json.loads((ROOT/f'reports/prepare/{si:04d}.json').read_text());rows+=d['rows'];errors+=d['errors']
    structure=ROOT/'reports/STRUCTURE_AUDIT.json';quarantined=set()
    if structure.exists():
        audit=json.loads(structure.read_text());assert audit['passed'];quarantined=set(audit['excluded_ids'])
        assert all(sha(ROOT/'reports/prepare'/name)==value for name,value in audit['prepare_receipts_sha'].items()),'structure audit drift'
    excluded,prov=external_families();groups=defaultdict(list);held=[]
    for r in rows:
        if r['i'] in quarantined:r['split']='structure_quarantine';held.append(r)
        elif r['family'] in excluded:r['split']='external_excluded';held.append(r)
        else:groups[r['scaffold']].append(r)
    total=sum(map(len,groups.values()));target=np.array([.85,.10,.05])*total;counts=np.zeros(3,int)
    # Greedy allocation, deterministic tie break; indivisible groups explain ratio deviations.
    for sc,g in sorted(groups.items(),key=lambda x:(-len(x[1]),hashlib.sha256(('20261118/'+x[0]).encode()).hexdigest())):
        candidates=[k for k in range(3) if counts[k]+len(g)<=target[k]]
        k=max(candidates,key=lambda k:target[k]-counts[k]) if candidates else int(np.argmax(target-counts))
        for r in g:r['split']=('train','val','test')[k]
        counts[k]+=len(g)
    rows.sort(key=lambda x:x['i']);family_sets={s:{r['family'] for r in rows if r['split']==s} for s in ('train','val','test')}
    assert not(family_sets['train']&family_sets['val'] or family_sets['train']&family_sets['test'] or family_sets['val']&family_sets['test'])
    atomic_json('manifest.json',rows)
    atomic_json('reports/data_contract.json',dict(source=str(SOURCE),source_sha256=SOURCE_SHA,source_records=limit,usable=len(rows),quarantined=len(errors),counts=Counter(r['split'] for r in rows),external_sources=prov,scaffold_groups=len(groups),duplicate_identities=len(rows)-len({r['canonical'] for r in rows}),node_qc=NODE_NAMES,edge_qc=EDGE_NAMES,coordinate_provenance='X_ref from source; per-record optimizer provenance unavailable',errors=errors))
    # Systemic errors must be fixed before proceeding; chemistry failures remain visible.
    if len(errors)>max(10,limit*.01):
        audit=ROOT/'reports/QUARANTINE_AUDIT.json'
        if not audit.exists():raise RuntimeError('More than 1% preparation failures: independent per-record audit required')
        accepted=json.loads(audit.read_text());assert accepted['passed'] and set(accepted['allowed_ids'])=={e['i'] for e in errors}
        assert all(sha(ROOT/'reports/prepare'/name)==value for name,value in accepted['prepare_receipts_sha'].items()),'quarantine audit drift'
    atomic_json('reports/PREPARED.json',dict(records=len(rows),source_records=limit,manifest_sha256=sha(ROOT/'manifest.json'),status='COMPLETE'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=301311);p.add_argument('--workers',type=int,default=16);p.add_argument('--finish-only',action='store_true');a=p.parse_args()
    if not a.finish_only:
        init()
        try:
            with concurrent.futures.ProcessPoolExecutor(a.workers,mp_context=mp.get_context('fork')) as pool:
                for d in pool.map(shard,[(s,a.limit) for s in range((a.limit+SHARD_SIZE-1)//SHARD_SIZE)]):print(json.dumps({k:v for k,v in d.items() if k not in ('rows','errors')}),flush=True)
        except BaseException as ex:
            atomic_json('reports/PREPARATION_FAILED.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
    finish(a.limit)
if __name__=='__main__':main()
