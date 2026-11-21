"""Tie-complete, batch-independent reference pooling for both GLI arms.

Adapted from slx-se3-clean-geometry 0792e071, not a checkpoint-compatible
claim. The learned layers and spherical/radial bases are unchanged.
"""
import math
import torch


def reference_matrix(dist, group, count, nearest=8, tie_band=1e-4):
    """Include all references tied at the per-edge kth distance boundary."""
    size=len(dist)
    degree=torch.bincount(group,minlength=count)
    width=int(degree.max()) if size else 0
    if width==0:return group.new_full((size,0),-1)
    # Sorting assigns storage slots only. No tied item is discarded by its slot.
    by_distance=torch.argsort(dist,stable=True)
    order=by_distance[torch.argsort(group[by_distance],stable=True)]
    start=degree.cumsum(0)-degree
    columns=torch.arange(size,device=group.device)-start[group[order]]
    table=group.new_full((count,width),size)
    table[group[order],columns]=order
    padded=torch.cat([dist,dist.new_full((1,),float('inf'))])
    sorted_dist=padded[table]
    k=(degree-1).clamp(min=0,max=nearest-1)
    boundary=sorted_dist[torch.arange(count,device=group.device),k]
    extra=(degree-1).clamp(min=0,max=nearest)
    next_boundary=sorted_dist[torch.arange(count,device=group.device),extra]
    threshold=torch.where(dist<=boundary[group]+tie_band,next_boundary[group],boundary[group])
    # The maximum includes all near-equal references, never a fixed tie cap.
    max_keep=int((sorted_dist<=next_boundary[:,None]+tie_band).sum(1).max())
    refs=table[group,:max_keep]
    valid=(refs<size)&(refs!=torch.arange(size,device=group.device)[:,None])
    valid=valid&(padded[refs]<=threshold[:,None]+tie_band)
    return torch.where(valid,refs,torch.full_like(refs,-1))


def install(module, scatter):
    """Keep the official parameter structure; replace only reference geometry."""
    module._feature_cache={}

    def pooled(self, vectors, dist, edges, z):
        source,target=edges
        refs_t=reference_matrix(dist,target,len(z))
        refs_s=reference_matrix(dist,source,len(z))
        temperature=max(float(self.cutoff)*.25,1e-6)
        eps=1e-12

        def angle_pool(refs, sign):
            rows=torch.arange(len(dist),device=dist.device)[:,None].expand_as(refs)
            valid=refs>=0;cur=rows[valid];ref=refs[valid]
            a=sign*vectors[cur];b=vectors[ref]
            cross=torch.linalg.cross(a,b)
            sine=cross.norm(dim=-1)/(a.norm(dim=-1)*b.norm(dim=-1)+eps)
            angle=torch.atan2(cross.norm(dim=-1),(a*b).sum(-1))
            w=torch.exp(-dist[ref]/temperature)*(sine>1e-4)
            feat=self.feature_expansion_2(dist[cur],angle)
            denom=scatter(w,cur,dim_size=len(dist),reduce='sum').clamp_min(eps)
            return scatter(feat*w[:,None],cur,dim_size=len(dist),reduce='sum')/denom[:,None]

        angles=.5*(angle_pool(refs_t,-1)+angle_pool(refs_s,1))
        width=refs_t.shape[1]
        pa,pb=torch.triu_indices(width,width,1,device=dist.device)
        features=[]
        # Bound temporary pair tensors without dropping any selected reference.
        chunk=max(1,65536//max(len(pa),1))
        for begin in range(0,len(dist),chunk):
            stop=min(len(dist),begin+chunk);refs=refs_t[begin:stop]
            ra,rb=refs[:,pa],refs[:,pb];valid=(ra>=0)&(rb>=0)
            rows=torch.arange(stop-begin,device=dist.device)[:,None].expand_as(ra)
            local=rows[valid];cur=local+begin;ra=ra[valid];rb=rb[valid]
            axis=-vectors[cur];va=vectors[ra];vb=vectors[rb]
            na=torch.linalg.cross(axis,va);nb=torch.linalg.cross(axis,vb)
            sine_a=na.norm(dim=-1)/(axis.norm(dim=-1)*va.norm(dim=-1)+eps)
            sine_b=nb.norm(dim=-1)/(axis.norm(dim=-1)*vb.norm(dim=-1)+eps)
            theta_a=torch.atan2(na.norm(dim=-1),(axis*va).sum(-1))
            theta_b=torch.atan2(nb.norm(dim=-1),(axis*vb).sum(-1))
            cosine=(na*nb).sum(-1)
            signed=(torch.linalg.cross(na,nb)*vectors[cur]).sum(-1)/(dist[cur]+eps)
            nondegenerate=(sine_a>1e-4)&(sine_b>1e-4)
            phi=torch.atan2(torch.where(nondegenerate,signed,torch.zeros_like(signed)),torch.where(nondegenerate,cosine,torch.ones_like(cosine)))
            # Smooth, antisymmetric orientation preference. Swapping the two
            # references swaps both features AND weights, including exact ties.
            preference=torch.tanh((dist[rb]-dist[ra])/.1+(z[source[rb]]-z[source[ra]]).to(dist)/4.)
            forward=self.feature_expansion_1(dist[cur],theta_a,phi)
            reverse=self.feature_expansion_1(dist[cur],theta_b,-phi)
            feat=.5*((1+preference[:,None])*forward+(1-preference[:,None])*reverse)
            w=torch.exp(-(dist[ra]+dist[rb])/temperature)*((sine_a>1e-4)&(sine_b>1e-4))
            denom=scatter(w,local,dim_size=stop-begin,reduce='sum').clamp_min(eps)
            features.append(scatter(feat*w[:,None],local,dim_size=stop-begin,reduce='sum')/denom[:,None])
        torsion=torch.cat(features) if features else dist.new_zeros((0,self.feature_expansion_1.num_radial*self.feature_expansion_1.num_spherical**2))
        return torsion,angles

    def forward(self,pos,z,batch):
        if self.emb.emb.weight.dtype==torch.float32:
            # Symmetric environments magnify FP32 roundoff through GraphNorm.
            # Parameters/Adam stay FP32; only this small geometric subnetwork
            # computes in FP64, with gradients returning through the casts.
            from torch.func import functional_call
            parameters={name:value.double() for name,value in self.named_parameters()}
            return functional_call(self,parameters,(pos.double(),z,batch)).float()
        pos=pos.double()
        edges=module.radius_graph(pos,r=self.cutoff,batch=batch)
        # Feature bases contain no trainable parameters and both official GMs
        # receive identical positions. Cache only for this Teacher forward.
        key=(pos.data_ptr(),z.data_ptr(),edges.data_ptr(),self.cutoff,
             self.feature_expansion_1.num_radial,self.feature_expansion_1.num_spherical)
        if key not in module._feature_cache:
            vectors=pos[edges[0]]-pos[edges[1]];dist=vectors.norm(dim=-1)
            module._feature_cache[key]=pooled(self,vectors,dist,edges,z)
        feature1,feature2=module._feature_cache[key]
        x=self.emb(z)
        # Run identical graph-local operations even for graphs with no edges.
        for block in self.interaction_blocks:x=block(x,feature1,feature2,edges,batch)
        for layer in self.lins:x=self.act(layer(x))
        return self.lin_out(x)

    module.GeometryModule_._forward=forward
