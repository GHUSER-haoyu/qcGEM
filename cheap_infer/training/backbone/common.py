"""Run-local paths and durable provenance, without modifying source data."""
import hashlib, json, os, pickle, time,io
from pathlib import Path
from collections import OrderedDict
ROOT=Path(os.environ.get('QCMOL_STUDENT_ROOT',Path(__file__).resolve().parent))
SOURCE=Path(os.environ.get('QCMOL_SOURCE_PT','/run/ti/20250101_processed.pt'))
CHECKPOINT=Path(os.environ.get('QCMOL_TEACHER_CKPT','/run/ti/qcgem_zenodo_17364257/qcGEM_ckpt_Index649.pt'))
SOURCE_SHA='44c929e30bfe2c07d3a8f7e6f777819930906468f8feddcc3ffdb17befacf5cc'
CHECKPOINT_SHA='ad87fb28b7e5a466eae31a0aa438df9c7248edf8cecf006803c327f337c0b59e'
UPSTREAM='8ba3ca3371002e2878a9f61e788df30fe260f037'
SEEDS=(101,211,307,401)
SHARD_SIZE=256
NODE_NAMES=[f'NAO_{i}' for i in range(6)]+[f'LP_{i}' for i in range(5)]+['LP_occupancy','ADCH','LI','NPA_charge','NPA_core','NPA_valence','NPA_rydberg']
EDGE_NAMES=[f'BD_{i}' for i in range(14)]+['BD_occupancy','Mayer','LBO','ELF','DI']
def atomic_json(path,value):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def event(path,**value):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(dict(time=time.time(),**value),allow_nan=False)+'\n')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def key(i):return f'{i:09d}'.encode()
class CacheUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        # NumPy 2 renamed its pickle module; training environment remains pinned
        # to the checkpoint-compatible NumPy 1 runtime. Array bytes are unchanged.
        if module.startswith('numpy._core'):module=module.replace('numpy._core','numpy.core',1)
        return super().find_class(module,name)
def cache_loads(data):return CacheUnpickler(io.BytesIO(data)).load()
def shard_path(kind,shard):return ROOT/'cache'/kind/f'{shard:04d}.lmdb'
def open_db(kind,shard,readonly=True):
    import lmdb
    p=shard_path(kind,shard)
    if not readonly:p.parent.mkdir(parents=True,exist_ok=True)
    return lmdb.open(str(p),map_size=64<<30,readonly=readonly,lock=not readonly,
                     readahead=False,meminit=False,max_readers=512,subdir=True)
def put(db,i,value):
    with db.begin(write=True) as t:t.put(key(i),pickle.dumps(value,protocol=5))
class Reader:
    def __init__(self,kind,max_open=32):
        self.kind=kind;self.dbs=OrderedDict();self.compat={};self.max_open=max_open
        path=ROOT/'reports/CHEAP_COMPATIBILITY.json'
        if kind=='cheap' and path.exists():self.compat=json.loads(path.read_text())['corrections']
    def get(self,i):
        s=int(i)//SHARD_SIZE
        if s not in self.dbs:
            if len(self.dbs)>=self.max_open:_,old=self.dbs.popitem(last=False);old.close()
            self.dbs[s]=open_db(self.kind,s)
        self.dbs.move_to_end(s)
        with self.dbs[s].begin() as t:b=t.get(key(i))
        if b is None:raise KeyError((self.kind,i))
        value=cache_loads(b)
        if self.kind=='cheap' and str(i) in self.compat:
            fix=self.compat[str(i)]
            for atom,col,new in fix['node_updates']:value['top']['g1'][atom,col]=new
            for edge,col,new in fix['edge_updates']:
                value['top']['e1_top'][edge,col]=new
                for view in value['views']:view['e1'][edge,col]=new
        return value
def prepared_shards(rank,world,watch=False):
    seen=set()
    while True:
        paths=[p for p in sorted((ROOT/'reports/prepare').glob('*.json')) if int(p.stem)%world==rank and p not in seen]
        for p in paths:seen.add(p);yield p
        # Re-scan files produced while the previous snapshot was being consumed.
        # Preparation can legitimately exit at an audit gate after all shards exist.
        if watch and paths:continue
        if not watch or (ROOT/'reports/PREPARED.json').exists():return
        if len(list((ROOT/'reports/prepare').glob('*.json')))==(301311+SHARD_SIZE-1)//SHARD_SIZE:return
        launch=ROOT/'reports/PREPARATION_LAUNCH.json'
        if launch.exists():
            pid=json.loads(launch.read_text())['pid']
            try:os.kill(pid,0)
            except ProcessLookupError:raise RuntimeError('CPU preparation exited without a complete manifest')
        time.sleep(10)
