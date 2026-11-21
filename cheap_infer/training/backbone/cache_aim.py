"""Pinned NSE, same model and geometry for Q,Q-1,Q+1; independent cheap feature cache."""
import argparse,json,time
import numpy as np
import torch
from aimnet.calculators import AIMNet2Calculator
from common import *
SUPPORTED={1,5,6,7,8,9,14,15,16,17,33,34,35,53}
MODEL_SHA='b98b3449b2a3765de6a188edff9344a38cc72df75f9d9bf68f4b898a253c5864'
def supported(r):
    z=r['top']['z'];rad=float(r['node'][:,22].sum());ne=int(z.sum()-r['node'][:,20].sum())
    return set(map(int,z))<=SUPPORTED and ((rad==0 and ne%2==0) or (rad==1 and ne%2==1))
def missing(r,status,dim=256):
    n=len(r['xyz']);return dict(node=np.zeros((n,dim+7),np.float32),edge=np.zeros((n*n,3),np.float32),global_=np.zeros(8,np.float32),forces=np.zeros((n,3),np.float32),energy_eV=None,latent_dim=dim,status=status)
def safe_forward(calc,items):
    try:return forward(calc,items)
    except (FloatingPointError,torch.cuda.OutOfMemoryError) as ex:
        torch.cuda.empty_cache()
        if len(items)>1:return [safe_forward(calc,[item])[0] for item in items]
        if isinstance(ex,torch.cuda.OutOfMemoryError):raise
        r,v=items[0]
        event('logs/aim_numerical_failures.jsonl',cid=r['cid'],error=str(ex),closest_distance=float(np.min(np.linalg.norm(v['xyz'][:,None]-v['xyz'][None,:],axis=-1)+np.eye(len(v['xyz']))*1e9)))
        return [missing(r,'nonfinite_single_molecule_AIM_result')]
def setup(device):
    path=ROOT/'assets/aimnet2nse_wb97m_0.pt'
    assert sha(path)==MODEL_SHA
    calc=AIMNet2Calculator(str(path),device=device,compile_model=False)
    calc.keys_out=list(calc.keys_out)+['aim'];calc.atom_feature_keys=list(calc.atom_feature_keys)+['aim']
    return calc
def forward(calc,items):
    dev=calc.device;b=len(items);nmax=max(len(r['xyz']) for r,v in items)
    coord=torch.zeros(b,nmax,3,device=dev);numbers=torch.zeros(b,nmax,dtype=torch.long,device=dev)
    charge=torch.zeros(b,device=dev);mult=torch.ones(b,device=dev)
    for k,(r,v) in enumerate(items):
        n=len(r['xyz']);coord[k,:n]=torch.as_tensor(v['xyz'],device=dev);numbers[k,:n]=torch.as_tensor(r['top']['z'],device=dev)
        charge[k]=float(r['node'][:,20].sum());mult[k]=1+float(r['node'][:,22].sum())
    base=calc(dict(coord=coord,numbers=numbers,charge=charge,mult=mult),forces=True)
    assert 'aim' in base,'Pinned AIM model did not expose aim latent'
    tri=(mult==1);ions=None
    if tri.any():
        ix=tri.nonzero().flatten();c=torch.cat([coord[ix],coord[ix]]);z=torch.cat([numbers[ix],numbers[ix]])
        q=torch.cat([charge[ix]-1,charge[ix]+1]);mm=torch.full_like(q,2.)
        ions=calc(dict(coord=c,numbers=z,charge=q,mult=mm),forces=False)
    tpos={int(i):j for j,i in enumerate(tri.nonzero().flatten().cpu())};nt=len(tpos);out=[]
    def arr(x):return x.detach().float().cpu().numpy()
    for k,(r,v) in enumerate(items):
        n=len(r['xyz']);q=arr(base['charges'][k,:n]);f=arr(base['forces'][k,:n]);latent=arr(base['aim'][k,:n]);energy=float(base['energy'][k].detach())
        assert latent.ndim==2
        plus=np.zeros(n,np.float32);minus=plus.copy();ip=ea=0.;has_tri=k in tpos
        if has_tri:
            j=tpos[k];qminus=arr(ions['charges'][j,:n]);qplus=arr(ions['charges'][j+nt,:n])
            plus=q-qminus;minus=qplus-q
            ip=float(ions['energy'][j+nt].detach())-energy;ea=energy-float(ions['energy'][j].detach())
        xyz=v['xyz'];center=xyz-xyz.mean(0);delta=xyz[:,None]-xyz[None,:];dist=np.linalg.norm(delta,axis=-1);off=~np.eye(n,dtype=bool)
        node=np.column_stack([np.ones(n),latent,q,np.linalg.norm(f,axis=1),np.full(n,has_tri),plus,minus,plus-minus]).astype(np.float32)
        edge=np.stack([np.ones((n,n)),q[:,None]*q[None,:]/(dist+.1)*off,((f[:,None]-f[None,:])*delta/(dist[:,:,None]+1e-6)).sum(-1)],-1).reshape(n*n,3).astype(np.float32)
        # Total E/N is stored explicitly; a train-only elemental baseline is subtracted during statistics fitting.
        glob=np.array([1,energy/n,np.linalg.norm((q[:,None]*center).sum(0)),has_tri,ip,ea,ip-ea,(ip-ea)/2],np.float32)
        for name,x in [('node',node),('edge',edge),('global',glob),('forces',f)]:
            if not np.isfinite(x).all():raise FloatingPointError(f'{r["cid"]}: nonfinite {name}')
        assert abs(float(q.sum())-float(charge[k]))<2e-3
        if has_tri:assert abs(float(plus.sum())-1)<3e-3 and abs(float(minus.sum())-1)<3e-3
        out.append(dict(node=node,edge=edge,global_=glob,forces=f,energy_eV=energy,latent_dim=latent.shape[1],status='ok'))
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=4);p.add_argument('--batch',type=int,default=128);p.add_argument('--watch',action='store_true');p.add_argument('--input');p.add_argument('--output');a=p.parse_args()
    torch.set_num_threads(1);calc=setup(f'cuda:{a.rank}');reader=Reader('cheap');dim=None;start=time.time();done=0
    if a.input:
        from pathlib import Path
        records=pickle.loads(Path(a.input).read_bytes());values=[None]*len(records);indices=[i for i,r in enumerate(records) if supported(r)]
        for lo in range(0,len(indices),a.batch):
            ii=indices[lo:lo+a.batch];out=safe_forward(calc,[(records[i],records[i]['views'][0]) for i in ii])
            for i,v in zip(ii,out):values[i]=v;dim=v['latent_dim']
        if dim is None:dim=256 # Verified dimension of this pinned checkpoint, never guessed from another model.
        for i,r in enumerate(records):
            if values[i] is None:
                n=len(r['xyz']);values[i]=dict(node=np.zeros((n,dim+7),np.float32),edge=np.zeros((n*n,3),np.float32),global_=np.zeros(8,np.float32),forces=np.zeros((n,3),np.float32),status='unsupported_elements_or_unknown_spin')
        Path(a.output).write_bytes(pickle.dumps(values,protocol=5));return
    for path in prepared_shards(a.rank,a.world,a.watch):
        si=int(path.stem)
        if (ROOT/f'reports/aim/{si:04d}.json').exists():continue
        rows=json.loads(path.read_text())['rows'];db=open_db('aim',si,False);pending=[];outputs={};records={};failures=[]
        def flush():
            nonlocal dim,done
            if not pending:return
            items=[(r,v) for i,j,r,v in pending]
            values=safe_forward(calc,items)
            for (i,j,r,v),value in zip(pending,values):
                dim=value['latent_dim'];outputs[i][j]=value
                if value['status']!='ok':failures.append(dict(i=i,view=j,reason=value['status']))
            pending.clear()
        for row in rows:
            r=reader.get(row['i']);i=r['i'];records[i]=r;outputs[i]=[None]*len(r['views'])
            if supported(r):
                for j,v in enumerate(r['views']):
                    pending.append((i,j,r,v))
                    if len(pending)>=a.batch:flush()
            else:failures.append(dict(i=i,reason='unsupported_elements_or_unknown_spin'))
        flush()
        if dim is None:raise RuntimeError('Cannot determine AIM latent dimension from supported molecules')
        for i,values in outputs.items():
            r=records[i];n=len(r['xyz'])
            for j,value in enumerate(values):
                if value is None:values[j]=dict(node=np.zeros((n,dim+7),np.float32),edge=np.zeros((n*n,3),np.float32),global_=np.zeros(8,np.float32),forces=np.zeros((n,3),np.float32),energy_eV=None,latent_dim=dim,status='unsupported_elements_or_unknown_spin')
            put(db,i,values);done+=1
        db.sync();db.close();atomic_json(f'reports/aim/{si:04d}.json',dict(status='COMPLETE',records=len(rows),latent_dim=dim,masked=failures,elapsed=time.time()-start))
        event(f'logs/aim_{a.rank}.jsonl',records=done,shard=si,elapsed=time.time()-start)
if __name__=='__main__':main()
