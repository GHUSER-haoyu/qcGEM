"""Cheap-only features. QC columns are never accessed by feature builders."""
import numpy as np
import time
import os,multiprocessing as mp
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors, rdEHTTools
from rdkit.Chem.EState import EStateIndices
from rdkit.Chem.Scaffolds import MurckoScaffold
from common import SEEDS
RDLogger.DisableLog('rdApp.warning')
BONDS=['UNSPECIFIED','SINGLE','DOUBLE','TRIPLE','QUADRUPLE','QUINTUPLE','HEXTUPLE','ONEANDAHALF','TWOANDAHALF','THREEANDAHALF','FOURANDAHALF','FIVEANDAHALF','AROMATIC','IONIC','HYDROGEN','THREECENTER','DATIVEONE','DATIVE','DATIVEL','DATIVER','OTHER','ZERO']
CHIRAL=['CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW','CHI_OTHER','CHI_TETRAHEDRAL','CHI_ALLENE','CHI_SQUAREPLANAR','CHI_TRIGONALBIPYRAMIDAL','CHI_OCTAHEDRAL']
FG_SMARTS=['[OX2H]','[NX3;H2,H1;!$(N-C=O)]','[NX3;H0;!$(N-C=O)]','C(=O)[OX2H1]','C(=O)[OX2H0]','C(=O)N','C(=O)[#6]','[CX3H1](=O)','[CX2]#N','[N+](=O)[O-]','S(=O)(=O)','S(=O)','[SX2H]','[SX2H0]','[nH]','[nH0]','[F,Cl,Br,I]','P(=O)','[CX3]=[CX3]','[CX2]#[CX2]']
FG=[Chem.MolFromSmarts(x) for x in FG_SMARTS]
EN={1:2.20,5:2.04,6:2.55,7:3.04,8:3.44,9:3.98,14:1.90,15:2.19,16:2.58,17:3.16,33:2.18,34:2.55,35:2.96,53:2.66}
IE={1:13.598,5:8.298,6:11.260,7:14.534,8:13.618,9:17.423,14:8.152,15:10.487,16:10.360,17:12.968,33:9.789,34:9.752,35:11.814,53:10.451}
PT=Chem.GetPeriodicTable()
def _eht_child(mol,send):
    try:
        fd=os.open(os.devnull,os.O_WRONLY);os.dup2(fd,1);os.dup2(fd,2);os.close(fd)
        ok,r=rdEHTTools.RunMol(mol,confId=0)
        if not ok:send.send((False,'RunMol returned false'));os._exit(0)
        value=(np.array(r.GetOrbitalEnergies(),copy=True),np.array(r.GetReducedChargeMatrix(),copy=True),
               int(r.numElectrons),np.array(r.GetAtomicCharges(),copy=True),np.array(r.GetReducedOverlapPopulationMatrix(),copy=True))
        send.send((True,value));send.close();os._exit(0)
    except BaseException as e:
        try:send.send((False,type(e).__name__+':'+str(e)))
        finally:os._exit(1)
def safe_eht(mol):
    # rdEHTTools can segfault after repeated calls. One native calculation per
    # disposable child contains that failure; the parent preserves the molecule.
    ctx=mp.get_context('fork');recv,send=ctx.Pipe(duplex=False);p=ctx.Process(target=_eht_child,args=(mol,send));p.start();send.close()
    try:
        if not recv.poll(20):raise RuntimeError('EHT timeout 20s')
        try:ok,value=recv.recv()
        except EOFError:raise RuntimeError('EHT native process exited without result')
        if not ok:raise RuntimeError(value)
        return value
    finally:
        recv.close();p.join(.2)
        if p.is_alive():p.terminate();p.join()
def identity(mol):
    heavy=Chem.RemoveHs(mol)
    canonical=Chem.MolToSmiles(heavy,isomericSmiles=True)
    family=nonstereo_identity(mol)
    plain=Chem.Mol(mol);Chem.RemoveStereochemistry(plain);plain=Chem.RemoveHs(plain)
    scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=plain,includeChirality=False)
    # The empty scaffold is kept as one group, not silently identity-split.
    return canonical,family,scaffold or '<ACYCLIC>'
def nonstereo_identity(mol):
    m=Chem.Mol(mol);Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(Chem.RemoveHs(m),isomericSmiles=False)
def set_xyz(mol,xyz):
    m=Chem.Mol(mol);m.RemoveAllConformers();c=Chem.Conformer(len(xyz))
    for i,p in enumerate(xyz):c.SetAtomPosition(i,[float(v) for v in p])
    m.AddConformer(c,assignId=True);return m
def reconstruct(node_cheap,edge_cheap,xyz):
    n=len(node_cheap);rw=Chem.RWMol();ec=edge_cheap.reshape(n,n,34)
    for row in node_cheap:
        a=Chem.Atom(int(round(row[0])));a.SetNoImplicit(True)
        a.SetFormalCharge(int(round(row[2])));a.SetNumRadicalElectrons(int(round(row[4])))
        a.SetChiralTag(getattr(Chem.ChiralType,CHIRAL[int(np.argmax(row[5:14]))]))
        a.SetIsAromatic(bool(row[59]));rw.AddAtom(a)
    for i in range(n):
        for j in range(i+1,n):
            bt=ec[i,j,12:34]
            if bt.sum()>0:rw.AddBond(i,j,getattr(Chem.BondType,BONDS[int(bt.argmax())]))
    m=rw.GetMol();Chem.SanitizeMol(m);m=set_xyz(m,xyz)
    # Bond insertion order differs from the original Mol. Derive tetrahedral tags
    # from mapped coordinates instead of reusing neighbor-order-dependent CW/CCW.
    Chem.AssignStereochemistryFrom3D(m,replaceExistingTags=True)
    Chem.AssignStereochemistry(m,cleanIt=True,force=True)
    assert m.GetNumAtoms()==n
    assert np.array_equal(np.array([a.GetAtomicNum() for a in m.GetAtoms()]),node_cheap[:,0])
    assert np.array_equal(np.array([a.GetFormalCharge() for a in m.GetAtoms()]),node_cheap[:,2])
    return m
def topology(mol,node_cheap,edge_cheap):
    n=mol.GetNumAtoms();z=np.array([a.GetAtomicNum() for a in mol.GetAtoms()],np.int16)
    # Recompute version-dependent cheap perception in the deployment RDKit.
    # The original Teacher tensors remain unchanged in each record.
    node_cheap=node_cheap.copy();hybrid=['S','SP','SP2','SP3','SP3D','SP3D2','OTHER','UNSPECIFIED']
    node_cheap[:,50:58]=0
    for a in mol.GetAtoms():node_cheap[a.GetIdx(),50+hybrid.index(str(a.GetHybridization()))]=1
    edge_cheap=edge_cheap.copy();ec=edge_cheap.reshape(n,n,34)
    for bond in mol.GetBonds():
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx();conj=float(bond.GetIsConjugated());ec[i,j,6:8]=ec[j,i,6:8]=[1-conj,conj]
    rings=np.array([[a.IsInRingSize(k) for k in range(3,9)] for a in mol.GetAtoms()],np.float32)
    cip=np.array([[a.HasProp('_CIPCode') and a.GetProp('_CIPCode')==x for x in ('R','S')] for a in mol.GetAtoms()],np.float32)
    fg=np.zeros((n,len(FG)),np.float32)
    for k,pat in enumerate(FG):
        for match in mol.GetSubstructMatches(pat):fg[list(match),k]=1
    constants=np.array([[EN.get(int(a),0),PT.GetRcovalent(int(a)),PT.GetRvdw(int(a)),PT.GetNOuterElecs(int(a)),IE.get(int(a),0),int(a in EN)] for a in z],np.float32)
    g1=np.concatenate([node_cheap,rings,cip,fg,constants],1)
    hop=np.minimum(Chem.GetDistanceMatrix(mol),9).astype(int)
    rot=np.zeros((n,n),np.float32)
    for i,j in mol.GetSubstructMatches(Chem.MolFromSmarts('[!$(*#*)&!D1]-!@[!$(*#*)&!D1]')):rot[i,j]=rot[j,i]=1
    ec=edge_cheap.reshape(n,n,34)
    e1=np.concatenate([ec,(ec[:,:,12:].sum(-1)==0)[:,:,None],rot[:,:,None],np.eye(10,dtype=np.float32)[hop]],2).reshape(n*n,-1)
    return dict(z=z,g1=g1.astype(np.float32),e1_top=e1.astype(np.float32),constants=constants)
def extract(mol,top):
    feature_start=time.time()
    n=mol.GetNumAtoms();xyz=np.asarray(mol.GetConformer().GetPositions(),np.float32);z=top['z']
    delta=xyz[:,None,:]-xyz[None,:,:];dist=np.linalg.norm(delta,axis=-1);off=~np.eye(n,dtype=bool)
    failures=[]
    mm=AllChem.MMFFGetMoleculeProperties(mol,mmffVariant='MMFF94');has_mm=mm is not None
    mq=np.array([mm.GetMMFFPartialCharge(i) for i in range(n)],np.float32) if has_mm else np.zeros(n,np.float32)
    energy=0.;energy_ok=False
    try:
        ff=AllChem.MMFFGetMoleculeForceField(mol,mm) if has_mm else AllChem.UFFGetMoleculeForceField(mol)
        if ff is not None:energy=float(ff.CalcEnergy());energy_ok=np.isfinite(energy)
    except Exception as e:failures.append('forcefield_energy:'+type(e).__name__)
    # MMFF assigns its own aromaticity model; restore RDKit chemical semantics.
    Chem.SanitizeMol(mol)
    gas=np.zeros(n,np.float32);gas_ok=False
    try:
        AllChem.ComputeGasteigerCharges(mol);gas=np.array([float(a.GetProp('_GasteigerCharge')) for a in mol.GetAtoms()],np.float32)
        gas_ok=bool(np.isfinite(gas).all());gas=np.nan_to_num(gas)
    except Exception as e:failures.append('gasteiger:'+type(e).__name__)
    estate=np.asarray(EStateIndices(mol),np.float32)
    cr=np.asarray(Crippen._GetAtomContribs(mol),np.float32)
    asa=np.asarray(rdMolDescriptors._CalcLabuteASAContribs(mol)[0],np.float32)
    phi=(mq[None,:]/(dist+.1)*off).sum(1)
    eh=np.zeros((n,3),np.float32);overlap=np.zeros((n,n),np.float32);levels=np.zeros(3,np.float32);has_eht=False
    try:
        en,rc,ne,charges,packed=safe_eht(mol)
        if True:
            h=(ne-1)//2;l=h+1
            assert rc.shape==(n,len(en)) and 0<=h<l<len(en)
            eh=np.stack([charges,rc[:,h],rc[:,l]],1).astype(np.float32)
            ii,jj=np.tril_indices(n)
            assert len(packed)==len(ii);overlap[ii,jj]=packed;overlap[jj,ii]=packed
            levels=np.array([en[h],en[l],en[l]-en[h]],np.float32)
            has_eht=bool(np.isfinite(eh).all() and np.isfinite(levels).all() and np.isfinite(overlap).all())
            if not has_eht:eh[:]=0;levels[:]=0;overlap[:]=0
    except Exception as e:failures.append('eht:'+type(e).__name__+':'+str(e)[:120])
    g2=np.column_stack([np.full(n,has_mm),mq,np.full(n,gas_ok),gas,estate,cr,asa,phi]).astype(np.float32)
    # EHT remains an independently masked group, never folded into generic columns.
    eht_node=np.column_stack([np.full(n,has_eht),eh]).astype(np.float32)
    c=top['constants'];vdw=(c[:,2,None]+c[None,:,2])/(dist+.1)*off
    empirical=np.stack([np.abs(c[:,0,None]-c[None,:,0]),vdw,mq[:,None]*mq[None,:]/(dist+.1)*off],-1).reshape(n*n,3)
    e1=np.concatenate([top['e1_top'],empirical],1).astype(np.float32)
    eht_edge=np.stack([np.full((n,n),has_eht),overlap],-1).reshape(n*n,2).astype(np.float32)
    center=xyz-np.average(xyz,axis=0,weights=[PT.GetAtomicWeight(int(a)) for a in z])
    u1=np.array([Descriptors.MolWt(mol),int((z>1).sum()),sum(PT.GetNOuterElecs(int(a)) for a in z)-Chem.GetFormalCharge(mol),Chem.GetFormalCharge(mol),rdMolDescriptors.CalcNumAromaticRings(mol),rdMolDescriptors.CalcTPSA(mol),Crippen.MolMR(mol),energy if energy_ok else 0,np.linalg.norm((mq[:,None]*center).sum(0)),has_mm,energy_ok],np.float32)
    result=dict(xyz=xyz,g2=g2,e1=e1,u1=u1,eht_node=eht_node,eht_edge=eht_edge,eht_global=np.r_[has_eht,levels].astype(np.float32),failures=failures)
    for k,v in result.items():
        if isinstance(v,np.ndarray) and not np.isfinite(v).all():raise ValueError('nonfinite cheap '+k)
    result['feature_seconds']=time.time()-feature_start
    return result
def build_views(mol,top):
    views=[extract(Chem.Mol(mol),top)];states=[]
    for seed in SEEDS:
        start=time.time()
        m=Chem.Mol(mol);m.RemoveAllConformers()
        result=AllChem.EmbedMolecule(m,randomSeed=seed,maxAttempts=50)
        retry=False
        if result<0:retry=True;result=AllChem.EmbedMolecule(m,randomSeed=seed,useRandomCoords=True,maxAttempts=50)
        if result<0:states.append(dict(seed=seed,status='embed_failed'));continue
        mm=AllChem.MMFFHasAllMoleculeParams(m)
        result=AllChem.MMFFOptimizeMolecule(m,mmffVariant='MMFF94',maxIters=200) if mm else AllChem.UFFOptimizeMolecule(m,maxIters=200)
        Chem.SanitizeMol(m)
        # Check that generated geometry respects explicit stereo, without reordering atoms.
        check=Chem.Mol(m);Chem.AssignStereochemistryFrom3D(check,replaceExistingTags=True)
        Chem.AssignStereochemistry(check,cleanIt=True,force=True)
        original={a.GetIdx():a.GetProp('_CIPCode') for a in mol.GetAtoms() if a.HasProp('_CIPCode')}
        if any(not check.GetAtomWithIdx(i).HasProp('_CIPCode') or check.GetAtomWithIdx(i).GetProp('_CIPCode')!=v for i,v in original.items()):
            states.append(dict(seed=seed,status='stereo_mismatch'));continue
        v=extract(m,top);v['seed']=seed;v['optimization_status']=int(result);views.append(v)
        states.append(dict(seed=seed,status='ok',retry=retry,forcefield='MMFF94' if mm else 'UFF',optimization_status=int(result),seconds=time.time()-start))
    if len(views)<2:raise ValueError('all independent conformers failed: '+str(states))
    return views,states
