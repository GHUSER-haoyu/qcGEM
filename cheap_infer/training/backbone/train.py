"""Four-rank DDP, bounded wall clock, atomic resume and validation-only selection."""
import argparse,contextlib,datetime,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from common import *
from data import Records,collate,device_batch,BucketBatches
from model import Student
from losses import compute
def save(path,state):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.pending');torch.save(state,tmp);os.replace(tmp,p)
def seed(value):random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
def batches_with_relations(dl,model,accum,epoch,rank,world,device,skip=0):
    if accum==1:
        yield from enumerate(dl);return
    import itertools
    iterator=iter(enumerate(dl))
    while True:
        group=list(itertools.islice(iterator,accum))
        if not group:return
        if group[-1][0]<skip:continue
        students=[];teachers=[]
        # A detached, identically seeded first pass keeps the relationship matrix
        # at the full effective batch size even when activations are accumulated.
        with torch.no_grad():
            for idx,b in group:
                seed(20261118+epoch*1000003+idx*17+rank);bb=device_batch(b,device)
                with torch.autocast('cuda',dtype=torch.bfloat16):o=model(bb,'gpu',decode=False)
                students.append(o['readout'].float());teachers.append(bb['t_readout'].float())
        sb=torch.cat(students);tb=torch.cat(teachers);ss=[torch.empty_like(sb) for _ in range(world)];tt=[torch.empty_like(tb) for _ in range(world)];dist.all_gather(ss,sb);dist.all_gather(tt,tb);sb_all=torch.cat(ss);tb_all=torch.cat(tt);offset=rank*len(sb)
        anchors=torch.tensor(sum(int(b['anchor'].sum()) for _,b in group),device=device);dist.all_reduce(anchors)
        for idx,b in group:
            b['relation_bank']=sb_all;b['relation_teacher_bank']=tb_all;b['relation_indices']=torch.arange(offset,offset+b['num_graphs'],device=device);b['geometry_bank_count']=anchors;b['geometry_group_size']=len(group);offset+=b['num_graphs'];yield idx,b
def evaluate(model,stats,rank,world,epoch,device):
    ds=Records('val',max(epoch,1));indices=list(range(rank,len(ds),world))
    dl=DataLoader(torch.utils.data.Subset(ds,indices),batch_size=4,collate_fn=collate,num_workers=1,pin_memory=True)
    model.eval();sums=dict(gpu=0.,cpu=0.);n=0
    with torch.no_grad():
        for b in dl:
            b=device_batch(b,device)
            for mode in sums:
                out=model(b,mode);_,_,score=compute(out,b,stats);sums[mode]+=float(score)*b['num_graphs']
            n+=b['num_graphs']
    atomic_json(f'validation/epoch_{epoch:03d}_rank_{rank}.json',dict(sums=sums,n=n))
    # Small numeric reduction only; no NCCL Python-object gathering.
    value=torch.tensor([sums['gpu'],sums['cpu'],n],device=device,dtype=torch.float64);dist.all_reduce(value)
    result=dict(gpu=float(value[0]/value[2]),cpu=float(value[1]/value[2]),records=int(value[2]));result['score']=(result['gpu']+result['cpu'])/2
    if rank==0:atomic_json(f'validation/epoch_{epoch:03d}.json',result)
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--microbatch',type=int,default=16);p.add_argument('--accum',type=int,default=1);p.add_argument('--max-hours',type=float,default=72);p.add_argument('--max-epochs',type=int,default=50);p.add_argument('--diagnostic-steps',type=int,default=0);a=p.parse_args()
    rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1));device=f'cuda:{local}'
    assert world==4,'Formal run requires four DDP ranks'
    torch.set_num_threads(1);torch.cuda.set_device(local);dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=60));seed(20261118)
    stats=json.loads((ROOT/'stats.json').read_text());model=Student(stats,device);model.set_stage(False)
    # Fixed optimizer groups retain their identity across freeze/unfreeze and resume.
    backbone=[];heads=[]
    for name,param in model.named_parameters():
        if not param.requires_grad:continue
        (backbone if name.startswith('core.encoder.') and not name.startswith('core.encoder.embedding_node_input.') else heads).append(param)
    opt=torch.optim.AdamW([dict(params=backbone,lr=8e-5,name='backbone'),dict(params=heads,lr=3e-4,name='heads')],weight_decay=1e-4)
    epoch0=0;batch0=0;step=0;best=float('inf');patience=0;refine_at=None;elapsed=0.;history=[]
    repair_path=ROOT/'reports/NUMERICAL_REPAIR.json'
    repair=json.loads(repair_path.read_text()) if repair_path.exists() else None
    if repair is not None:assert repair['passed'],'Numerical repair gate is closed'
    revision=repair['revision'] if repair is not None else 'initial'
    revision_boundary=False;inherited_mixed_epoch=False
    last=ROOT/'checkpoints/last.pt'
    if last.exists():
        st=torch.load(last,weights_only=False,map_location='cpu');model.load_state_dict(st['model']);opt.load_state_dict(st['optimizer'])
        epoch0=st['epoch_next'];step=st['step'];best=st['best'];patience=st['patience'];refine_at=st['refine_at'];elapsed=st['elapsed'];history=st['history']
        batch0=st.get('batch_next',0)
        assert st['manifest_sha']==sha(ROOT/'manifest.json') and st['stats_sha']==sha(ROOT/'stats.json'),'resume inputs drift'
        revision_boundary=st.get('numerical_revision','initial')!=revision
        inherited_mixed_epoch=st.get('mixed_numerical_epoch',False)
        if revision_boundary:patience=0;refine_at=None
    if not a.diagnostic_steps:
        gates=json.loads((ROOT/'reports/ACCEPTANCE.json').read_text());assert gates['passed'],'Acceptance gate is closed'
    start=time.time()
    if rank==0 and not (ROOT/'reports/TRAINING_START.json').exists():
        atomic_json('reports/TRAINING_START.json',dict(start_unix=start,deadline_unix=start+a.max_hours*3600,max_hours=a.max_hours,diagnostic=bool(a.diagnostic_steps)))
    dist.barrier();deadline=json.loads((ROOT/'reports/TRAINING_START.json').read_text())['deadline_unix']
    # DDP constructed with all potential parameters active; warmup masks gradients via lr=0.
    ddp=DDP(model,device_ids=[local],find_unused_parameters=True,broadcast_buffers=False)
    for epoch in range(epoch0,a.max_epochs):
        mixed_epoch=epoch==epoch0 and (inherited_mixed_epoch or revision_boundary and batch0>0)
        warm=epoch==0;model.set_stage(warm);model.train()
        ds=Records('train',epoch);sampler=BucketBatches(ds,rank,world,a.microbatch,epoch)
        dl=DataLoader(ds,batch_sampler=sampler,collate_fn=collate,num_workers=2,pin_memory=True,persistent_workers=False)
        opt.zero_grad(set_to_none=True);seen=0;epoch_start=time.time();stopped=False
        for batch_idx,b in batches_with_relations(dl,model,a.accum,epoch,rank,world,device,batch0 if epoch==epoch0 else 0):
            if epoch==epoch0 and batch_idx<batch0:continue
            # Per-rank deterministic masks also make mid-epoch resume reproducible.
            seed(20261118+epoch*1000003+batch_idx*17+rank)
            b=device_batch(b,device);progress=(epoch+batch_idx/max(len(dl),1))/a.max_epochs
            if warm:head_lr=5e-4*min(1.,(batch_idx+1)/max(len(dl),1));core_lr=0.
            elif refine_at is not None:head_lr=core_lr=2e-5-(2e-5-5e-6)*min(1.,(epoch-refine_at+batch_idx/max(len(dl),1))/2)
            else:factor=.1+.9*(1+math.cos(math.pi*progress))/2;head_lr=3e-4*factor;core_lr=8e-5*factor
            opt.param_groups[0]['lr']=core_lr;opt.param_groups[1]['lr']=head_lr
            sync=(batch_idx+1)%a.accum==0 or batch_idx+1==len(dl)
            context=contextlib.nullcontext() if sync else ddp.no_sync()
            # Last accumulation group is normalized by its actual number of microbatches.
            remaining=len(dl)-(batch_idx//a.accum)*a.accum;denom=min(a.accum,remaining)
            with context:
                with torch.autocast('cuda',dtype=torch.bfloat16):out=ddp(b,'gpu')
                weight=.1 if warm else min(.5,.1+.2*epoch)
                loss,parts,score=compute(out,b,stats,weight)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
                (loss/denom).backward()
            seen+=b['num_graphs']*world
            if sync:
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                if warm:
                    # Weight decay must not change frozen backbone during warmup.
                    for p0 in backbone:p0.grad=None
                opt.step();opt.zero_grad(set_to_none=True);step+=1
                if step%200==0:
                    if rank==0:
                        save('checkpoints/last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch_next=epoch,batch_next=batch_idx+1,step=step,best=best,patience=patience,refine_at=refine_at,elapsed=elapsed+time.time()-start,history=history,manifest_sha=sha(ROOT/'manifest.json'),stats_sha=sha(ROOT/'stats.json'),diagnostic=bool(a.diagnostic_steps),resume_at_optimizer_boundary=True,numerical_revision=revision,mixed_numerical_epoch=mixed_epoch))
                    dist.barrier()
                if rank==0 and (step%20==0 or step==1):
                    event('logs/train.jsonl',epoch=epoch,step=step,loss=float(loss),parts={k:float(v.detach()) for k,v in parts.items()},seen_epoch=seen,elapsed=elapsed+time.time()-start,molecules_per_second=seen/max(time.time()-epoch_start,1e-6),grad_norm=float(norm),microbatch=a.microbatch,accum=a.accum)
                stop=torch.tensor(int(time.time()>=deadline or a.diagnostic_steps and step>=a.diagnostic_steps),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
                if stop.item():stopped=True;break
        val=evaluate(model,stats,rank,world,epoch,device)
        meaningful=val['score']<best*(1-.001);improved=val['score']<best
        patience=0 if meaningful else patience+1
        if improved:best=val['score']
        history.append(dict(epoch=epoch,validation=val,complete_epoch=not stopped,exposure_this_process=seen,unique_records=len(ds),padded_exposures=sampler.padded,numerical_revision=revision,numerically_complete_epoch=not stopped and not mixed_epoch))
        eligible_joint=sum(h['epoch']>=1 and h.get('numerical_revision','initial')==revision and h.get('numerically_complete_epoch',h['complete_epoch']) for h in history)
        if eligible_joint>=6 and patience>=5 and refine_at is None:refine_at=epoch+1
        if rank==0:
            state=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch_next=epoch+1,step=step,best=best,patience=patience,refine_at=refine_at,elapsed=elapsed+time.time()-start,history=history,manifest_sha=sha(ROOT/'manifest.json'),stats_sha=sha(ROOT/'stats.json'),diagnostic=bool(a.diagnostic_steps),resume_at_epoch_boundary=True,incomplete_epoch=stopped,numerical_revision=revision,mixed_numerical_epoch=False)
            save('checkpoints/last.pt',state)
            if improved:save('checkpoints/best.pt',state)
            atomic_json('reports/TRAINING_STATE.json',dict(status='RUNNING',epoch=epoch,step=step,validation=val,elapsed=state['elapsed'],best=best))
        dist.barrier()
        if stopped or refine_at is not None and epoch>=refine_at+1:break
    if rank==0:atomic_json('reports/TRAINING_ENDED.json',dict(status='TRAINING_ENDED_EVALUATION_PENDING',step=step,epochs=len(history),elapsed=elapsed+time.time()-start,checkpoint=str(last),reason='deadline_or_convergence_or_epoch_cap'))
    dist.destroy_process_group()
if __name__=='__main__':main()
