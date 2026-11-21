"""Original Teacher and separately instantiated geometry-corrected Student core."""
import ast, importlib.util, sys, types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data,Batch
from common import CHECKPOINT
SRC=Path(__file__).resolve().parent/'vendor'
sys.path.insert(0,str(SRC))
def load_module(stable=False):
    name='_qcgem_stable' if stable else '_qcgem_original'
    if name in sys.modules:return sys.modules[name],sys.modules[name+'_geo']
    spec=importlib.util.spec_from_file_location(name+'_geo',SRC/'GeometryModule.py')
    geo=importlib.util.module_from_spec(spec);sys.modules[spec.name]=geo;spec.loader.exec_module(geo)
    tree=ast.parse((SRC/'models.py').read_text())
    tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.module=='GeometryModule')]
    m=types.ModuleType(name);m.__file__=str(SRC/'models.py');m.GeometryModule_=geo.GeometryModule_;sys.modules[name]=m
    exec(compile(tree,m.__file__,'exec'),m.__dict__)
    original=m.MaskProcess
    def clean(*a,**kw):kw['noise_size']=0.;return original(*a,**kw)
    m.MaskProcess=clean
    if stable:
        import stable_ops as s
        from symmetric_geometry import install
        geo.scatter=s.deterministic_scatter;geo.scatter_min=s.stable_min;geo.EdgeGraphConv.aggregate=s.stable_aggregate
        install(geo,s.deterministic_scatter)
        m.softmax=s.stable_softmax;m.qcGEM_Transformer.aggregate=s.stable_aggregate
    return m,geo
def create(device='cpu',stable=False,load_weights=True):
    m,geo=load_module(stable)
    model=m.qcGEM(input_global_dim=[200,512,512],global_head_dim=32,botnec_global_dim=128,
       input_node_dim=80,node_head_dim=32,BotNec_node_dim=128,input_edge_dim=53,edge_head_dim=32,BotNec_edge_dim=128,
       heads=8,device=device,act_fn=nn.GELU(),norm='layer',remove_self_loop=False,
       global_mask_ratio=1.,mask_ratio=0.,replace_ratio=0.,remask_ratio=0.,
       encoder_method='qcGEM_Encoder',decoder_method='qcGEM_Decoder',encoder_layers=16,decoder_layers=0,
       gm_cutoff=8.,gm_output_dim=12,gm_interact_time=4,gm_layer_num=3)
    if load_weights:
        state=torch.load(CHECKPOINT,weights_only=False,map_location='cpu')['model_state_dict'];model.load_state_dict(state,strict=True)
    if stable:
        from stable_ops import stable_graph_norm
        from stable_ops import deterministic_scatter
        def stable_mean(self,x,index=None,ptr=None,dim_size=None,dim=-2,**kwargs):
            assert index is not None and dim in (-2,0)
            return deterministic_scatter(x,index,dim_size=dim_size,reduce='mean')
        for block in model.encoder.modules():
            if isinstance(block,m.aggr.MeanAggregation):block.forward=types.MethodType(stable_mean,block)
        for g in (model.encoder.GM_1,model.encoder.GM_2):
            for block in g.interaction_blocks:block.norm_layer.forward=types.MethodType(stable_graph_norm,block.norm_layer)
    return model,geo
def official_batch(records,views,device):
    out=[]
    for r,v in zip(records,views):
        n=len(r['xyz']);ii=np.repeat(np.arange(n),n);jj=np.tile(np.arange(n),n)
        out.append(Data(num_nodes=n,CID=r['cid'],xyz=torch.tensor(v['xyz'],dtype=torch.float32),
            node_features=torch.tensor(r['node'],dtype=torch.float32),edge_features=torch.tensor(r['edge'],dtype=torch.float32),
            edge_index=torch.tensor(np.array([ii,jj]),dtype=torch.long),
            global_MD=torch.tensor(np.nan_to_num(r['md'])[None],dtype=torch.float32),global_FP=torch.tensor(r['fp'][None],dtype=torch.float32)))
    return Batch.from_data_list(out).to(device)
def encode(model,b):
    d,label=model.pre_process(b);d,states=model.encoder(d)
    readout=model.encoder.pool_graph_embedding(d,states)
    return d,states,readout,label
def split_teacher(d,states,readout):
    ptr=d['ptr'].cpu().tolist();edge=d['edge_index'];result=[]
    for k,(lo,hi) in enumerate(zip(ptr[:-1],ptr[1:])):
        mask=(edge[0]>=lo)&(edge[0]<hi)
        get=lambda x:x.detach().float().cpu().numpy()
        item=dict(node=get(d['node_features'][lo:hi]),edge=get(d['edge_features'][mask]),global_=get(d['global_features'][k]),readout=get(readout[k]),
           mid_node=np.stack([get(states['Node'][l][lo:hi]) for l in (1,2,3)]),mid_global=np.stack([get(states['Global'][l][k]) for l in (1,2,3)]))
        result.append(item)
    return result
