import torch
from torch.nn import functional as F
from model import mean
def graph_mean(x,idx,n):return mean(x,idx,n).mean()
def normalized(x):return F.layer_norm(x.float(),(x.shape[-1],))
def compute(out,b,stats,decoder_weight=.5):
    ng=b['num_graphs'];nb=b['node_batch'];eb=b['edge_batch'];loss={}
    def scale(key):return torch.as_tensor(stats['teacher'][key]['std'],device=nb.device)
    def align(s,t,k,index=None,weight=None):
        s=s.float();t=t.float();l=F.smooth_l1_loss(s/scale(k),t/scale(k),reduction='none').mean(-1)+.5*(1-F.cosine_similarity(s,t,dim=-1))
        if index is None:return l.mean()
        if weight is not None:return (mean(l*weight,index,ng)/mean(weight,index,ng).clamp_min(1e-8)).mean()
        return graph_mean(l,index,ng)
    i,j=b['edge_index'];dist=(b['reference_xyz'][i]-b['reference_xyz'][j]).norm(dim=-1)
    cov=(b['raw_edge'][:,31:53].sum(-1)>0).float();ew=cov+torch.exp(-dist/4)
    loss['lat_node']=align(out['node'],b['t_node'],'node',nb)
    loss['lat_edge']=.5*align(out['edge'],b['t_edge'],'edge',eb,ew)
    loss['lat_global']=align(out['global_'],b['t_global_'],'global_')
    loss['readout']=align(out['readout'],b['t_readout'],'readout')
    loss['inter']=.2*(graph_mean(F.smooth_l1_loss(normalized(out['mid_node']),normalized(b['t_mid_node']),reduction='none').mean((0,2)),nb,ng)+F.smooth_l1_loss(normalized(out['mid_global']),normalized(b['t_mid_global'])))
    s=F.normalize(out['readout'].float(),dim=-1);t=F.normalize(b['t_readout'].float(),dim=-1)
    if 'relation_bank' in b:
        bank=F.normalize(b['relation_bank'].float(),dim=-1);target_bank=F.normalize(b['relation_teacher_bank'].float(),dim=-1)
        terms=F.smooth_l1_loss(s@bank.T,t@target_bank.T,reduction='none');off=torch.ones_like(terms,dtype=torch.bool);off[torch.arange(len(s),device=nb.device),b['relation_indices']]=False
        local=terms[off].mean();loss['relation']=.1*(2*local-local.detach())
    elif torch.is_grad_enabled() and torch.distributed.is_initialized():
        from torch.distributed.nn.functional import all_gather
        s=torch.cat(all_gather(s));tt=[torch.empty_like(t) for _ in range(torch.distributed.get_world_size())];torch.distributed.all_gather(tt,t);t=torch.cat(tt)
    if 'relation_bank' not in b:
        relation=F.smooth_l1_loss(s@s.T,t@t.T,reduction='none');off=~torch.eye(len(s),dtype=torch.bool,device=nb.device)
        loss['relation']=.1*relation[off].mean() if len(s)>1 else s.sum()*0
    dec=out['decoded'];nq=b['raw_node'][:,:18].float();eq=b['raw_edge'][:,:19].float()
    def physical_families(terms,index,families):
        return torch.stack([graph_mean(terms[:,lo:hi].mean(-1),index,ng) for lo,hi in families]).mean()
    nf=[(0,6),(6,12),(12,13),(13,14),(14,18)];ef=[(0,15),(15,16),(16,17),(17,18),(18,19)]
    # Native normalized QC reconstruction and additional raw-unit recovery are separate.
    nqnorm=normalized(nq);eqnorm=normalized(eq)
    loss['native_qc']=physical_families((dec['node_features'][:,:18]-nqnorm).abs(),nb,nf)+physical_families((dec['edge_features'][:,:19]-eqnorm).abs(),eb,ef)
    ns=torch.tensor(stats['node_qc']['std'],device=nb.device);es=torch.tensor(stats['edge_qc']['std'],device=nb.device)
    loss['raw_qc']=physical_families(F.smooth_l1_loss(out['raw_node']/ns,nq/ns,reduction='none'),nb,nf)+physical_families(F.smooth_l1_loss(out['raw_edge']/es,eq/es,reduction='none'),eb,ef)
    affine=[]
    for pred,y,idx in ((out['node_affine'],nq,nb),(out['edge_affine'],eq,eb)):
        mu=y.mean(-1);sigma=(y.var(-1,unbiased=False)+1e-5).sqrt()
        affine.append(graph_mean(F.smooth_l1_loss(pred[:,0],mu,reduction='none')+F.smooth_l1_loss(F.softplus(pred[:,1])+1e-5,sigma,reduction='none'),idx,ng))
    loss['affine']=.1*sum(affine)
    cats=[]
    for pred,label,parts,idx in [(dec['node_features'],b['raw_node'],[(23,32),(32,68),(68,76),(76,78),(78,80)],nb),(dec['edge_features'],b['raw_edge'],[(19,25),(25,27),(27,29),(29,31),(31,53)],eb)]:
        for lo,hi in parts:
            valid=label[:,lo:hi].sum(-1)>0
            terms=F.cross_entropy(pred[:,lo:hi],label[:,lo:hi].argmax(-1),reduction='none')*valid
            cats.append((mean(terms,idx,ng)/mean(valid.float(),idx,ng).clamp_min(1e-8)).mean())
    loss['topology']=.1*sum(cats)/len(cats)
    loss['global_targets']=.1*(F.l1_loss(dec['global_MD'],b['md'])+F.binary_cross_entropy_with_logits(dec['global_FP'],b['fp']))
    # The folding decoder creates coordinates in its own frame. Align with a proper rotation only.
    geom=[]
    for g in range(ng):
        if not bool(b['anchor'][g]):continue
        lo,hi=b['ptr'][g:g+2].tolist();p=dec['xyz'][:,lo:hi].float();y=b['reference_xyz'][lo:hi].float()
        pc=p-p.mean(1,keepdim=True);yc=y-y.mean(0,keepdim=True)
        with torch.no_grad():
            u,_,vh=torch.linalg.svd(pc.detach().transpose(1,2)@yc.expand_as(pc));fix=torch.eye(3,device=p.device).repeat(len(p),1,1);fix[:,2,2]=torch.det(u@vh);rot=u@fix@vh
        pa=pc@rot;diff=pa-yc
        rms=(diff.square().sum(-1).mean(-1)+1e-8).sqrt().mean()
        pd=(torch.cdist(p,p)-torch.cdist(y,y)[None]).abs().mean()
        pv=(diff[:,:,None]-diff[:,None,:]).abs().mean()
        # Deterministic noncollinear atom frames; no mirror alignment.
        frames=[]
        for a in range(min(hi-lo,8)):
            ds=(yc-yc[a]).norm(dim=-1);order=torch.argsort(ds);near=order[1:]
            if len(near)<2:continue
            x=yc[near[0]]-yc[a];ok=torch.cross(x.expand_as(yc[near[1:]]),yc[near[1:]]-yc[a],dim=-1).norm(dim=-1)>1e-3
            if ok.any():frames.append([a,int(near[0]),int(near[1:][ok][0])])
        fape=p.sum()*0
        if frames:
            from utils import local_coord_shoot
            f=torch.tensor(frames,device=p.device);lp=local_coord_shoot(f,p);ly=local_coord_shoot(f,y[None]);fape=(lp-ly).norm(dim=-1).clamp_max(5).mean()
        geom.append(rms+pd+pv+fape)
    geometry_sum=torch.stack(geom).sum() if geom else out['node'].sum()*0
    count=torch.tensor(float(len(geom)),device=nb.device)
    world=1
    if 'geometry_bank_count' in b:
        count=b['geometry_bank_count'];world=torch.distributed.get_world_size()*b['geometry_group_size']
    elif torch.is_grad_enabled() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(count);world=torch.distributed.get_world_size()
    loss['geometry']=geometry_sum*world/count.clamp_min(1)
    native=sum(loss[k] for k in ('native_qc','topology','global_targets','geometry'))
    total=sum(loss[k] for k in ('lat_node','lat_edge','lat_global','readout','inter','relation','raw_qc','affine'))+decoder_weight*native
    # Validation selection consists only of independently measurable QC and readout errors.
    score=.5*(graph_mean(((out['raw_node']-nq)/ns).square().mean(-1),nb,ng)+graph_mean(((out['raw_edge']-eq)/es).square().mean(-1),eb,ng))
    score=.5*(score+((out['readout'].float()-b['t_reference_readout'])/scale('readout')).square().mean())
    return total,loss,score
