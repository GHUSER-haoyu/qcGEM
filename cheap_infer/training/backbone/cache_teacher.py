"""Four independent GPU workers cache the frozen official Teacher once per view."""
import argparse,json,os,time
import numpy as np
import torch
from common import *
from official import create,official_batch,encode,split_teacher
def main():
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=4);p.add_argument('--watch',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);gpu=a.rank%torch.cuda.device_count();torch.cuda.set_device(gpu);device=f'cuda:{gpu}'
    model,_=create(device);model.eval().requires_grad_(False)
    reader=Reader('cheap');start=time.time();done=0
    for pth in prepared_shards(a.rank,a.world,a.watch):
        si=int(pth.stem);receipt=ROOT/f'reports/teacher/{si:04d}.json'
        if receipt.exists() and json.loads(receipt.read_text()).get('prepare_sha')==sha(pth):continue
        rows=json.loads(pth.read_text())['rows'];db=open_db('teacher',si,False)
        for row in rows:
            with db.begin() as txn:exists=txn.get(key(row['i'])) is not None
            if exists:continue
            r=reader.get(row['i']);n=len(r['xyz']);targets=[]
            # All views for one molecule batch together: fixed feature values, different geometry.
            with torch.no_grad():
                d,s,h,l=encode(model,official_batch([r]*len(r['views']),r['views'],device))
                targets=split_teacher(d,s,h)
            maxerr=0.
            for t in targets:
                for k,x in t.items():
                    y=x.astype(np.float16)
                    if not np.isfinite(y).all():raise ValueError('Teacher cache float16 overflow')
                    err=np.linalg.norm(y.astype(np.float32)-x)/max(np.linalg.norm(x),1e-8);maxerr=max(maxerr,float(err))
                    if err>1e-3:raise ValueError('Teacher FP16 cache precision gate')
                    t[k]=y
            put(db,r['i'],dict(targets=targets,max_relative_fp16_error=maxerr,reference_view=0,other_views='frozen_QC_geometry_response_not_new_QC_truth'))
            done+=1
            if done%32==0:event(f'logs/teacher_{a.rank}.jsonl',records=done,elapsed=time.time()-start,shard=si,last=r['i'])
        db.sync();db.close();atomic_json(f'reports/teacher/{si:04d}.json',dict(status='COMPLETE',records=len(rows),rank=a.rank,prepare_sha=sha(pth)))
if __name__=='__main__':main()
