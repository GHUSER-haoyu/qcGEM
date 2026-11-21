import json,pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from common import ROOT,Reader
GROUPS=('g1','g2','eht_node','aim_node','e1','eht_edge','aim_edge','u1','eht_global','aim_global')
def groups(r,v,a):
    g1=np.asarray(r['top']['g1'],np.float32).copy()
    # CW/CCW refers to neighbor insertion order, not an invariant molecular
    # property. Canonicalize this student-only block to unspecified/R/S.
    cip=g1[:,68:70].copy();g1[:,5:14]=0;g1[:,5]=cip.sum(-1)==0;g1[:,6:8]=cip
    return dict(g1=g1,g2=v['g2'],eht_node=v['eht_node'],aim_node=a['node'],e1=v['e1'],eht_edge=v['eht_edge'],aim_edge=a['edge'],u1=v['u1'][None],eht_global=v['eht_global'][None],aim_global=a['global_'][None])
class Records(Dataset):
    def __init__(self,split,epoch=0):
        self.rows=[r for r in json.loads((ROOT/'manifest.json').read_text()) if r['split']==split];self.epoch=epoch;self.split=split;self.readers=None
    def __len__(self):return len(self.rows)
    def __getitem__(self,k):
        if self.readers is None:self.readers=[Reader(x) for x in ('cheap','aim','teacher')]
        row=self.rows[k];r,a,t=[reader.get(row['i']) for reader in self.readers]
        # Reference warmup. Later 10% anchor views, otherwise deterministic independent views.
        if self.epoch==0:j=0
        elif self.split=='train':
            gen=np.random.default_rng(row['i']+1000003*self.epoch)
            j=0 if gen.random()<.1 else int(gen.integers(1,len(r['views'])))
        else:j=1
        return dict(record=r,view=r['views'][j],aim=a[j],teacher=t['targets'][j],reference=t['targets'][0],view_index=j)
def collate(items):
    b={k:[] for k in GROUPS};nb=[];eb=[];edges=[];xyz=[];forces=[];nodes=[];rawedges=[];z=[];ptr=[0];eptr=[0];gs=[]
    teacher={k:[] for k in ('node','edge','global_','readout','mid_node','mid_global')};reference=[]
    for g,it in enumerate(items):
        r,v,a,t=it['record'],it['view'],it['aim'],it['teacher'];n=len(r['xyz']);offset=ptr[-1]
        for k,x in groups(r,v,a).items():b[k].append(np.asarray(x,np.float32))
        for k in teacher:teacher[k].append(np.asarray(t[k],np.float32))
        edges.append(np.array([np.repeat(np.arange(n),n),np.tile(np.arange(n),n)])+offset)
        nb.append(np.full(n,g));eb.append(np.full(n*n,g));ptr.append(offset+n);eptr.append(eptr[-1]+n*n)
        xyz.append(v['xyz']);forces.append(a['forces']);nodes.append(r['node']);rawedges.append(r['edge']);z.append(r['top']['z']);reference.append(r['xyz'])
    b={k:torch.from_numpy(np.concatenate(v)) for k,v in b.items()}
    for k,vs in teacher.items():
        if k=='mid_node':arr=np.concatenate(vs,axis=1)
        elif k in ('global_','readout','mid_global'):arr=np.stack(vs)
        else:arr=np.concatenate(vs)
        b['t_'+k]=torch.from_numpy(arr)
    b.update(xyz=torch.tensor(np.concatenate(xyz)),forces=torch.tensor(np.concatenate(forces)),raw_node=torch.tensor(np.concatenate(nodes)),raw_edge=torch.tensor(np.concatenate(rawedges)),
       z=torch.tensor(np.concatenate(z),dtype=torch.long),edge_index=torch.tensor(np.concatenate(edges,1),dtype=torch.long),node_batch=torch.tensor(np.concatenate(nb),dtype=torch.long),edge_batch=torch.tensor(np.concatenate(eb),dtype=torch.long),
       ptr=torch.tensor(ptr),eptr=torch.tensor(eptr),reference_xyz=torch.tensor(np.concatenate(reference)),anchor=torch.tensor([it['view_index']==0 for it in items]),
       md=torch.tensor(np.stack([np.nan_to_num(it['record']['md']) for it in items])),fp=torch.tensor(np.stack([it['record']['fp'] for it in items])),
       ids=[it['record']['i'] for it in items],cids=[it['record']['cid'] for it in items],num_graphs=len(items))
    b['t_reference_readout']=torch.tensor(np.stack([it['reference']['readout'].astype(np.float32) for it in items]))
    b['t_reference_node']=torch.tensor(np.concatenate([it['reference']['node'].astype(np.float32) for it in items]))
    return b
def device_batch(b,device):return {k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in b.items()}
class BucketBatches:
    def __init__(self,ds,rank,world,micro,epoch):
        rng=np.random.default_rng(20261118+epoch);ids=rng.permutation(len(ds)).tolist();ordered=[]
        for k in range(0,len(ids),1024):ordered.extend(sorted(ids[k:k+1024],key=lambda i:ds.rows[i]['n']))
        width=world*micro;pad=(-len(ordered))%width
        if pad:ordered+=ordered[:pad]
        batches=[ordered[k:k+width] for k in range(0,len(ordered),width)];rng.shuffle(batches)
        self.batches=[b[rank*micro:(rank+1)*micro] for b in batches];self.padded=pad
    def __len__(self):return len(self.batches)
    def __iter__(self):return iter(self.batches)
