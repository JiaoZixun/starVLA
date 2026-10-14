#!/usr/bin/env python3
"""Build and validate RoboDojo FK geometry sidecars without copying the dataset."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import xml.etree.ElementTree as ET

DEFAULT_DATA = Path('/home/data/robodojo_cache/starvla_robodojo_v30/RoboDojo_lerobot_v30_video')
DEFAULT_URDF = Path('/home/data/roboskills/RoboDojo/.cache/robodojo_assets_repo/Assets/Robots/x5/X5A.urdf')

def _transform(xyz, rpy):
    t = np.eye(4, dtype=np.float32); t[:3,:3] = Rotation.from_euler('xyz', rpy).as_matrix(); t[:3,3] = xyz
    return t

def load_chain(urdf):
    root = ET.parse(urdf).getroot(); chain=[]
    parents = {j.find('child').attrib['link']: j for j in root.findall('joint') if j.find('child') is not None}
    link='link6'
    while link != 'base_link':
        j=parents[link]; o=j.find('origin'); a=j.find('axis')
        chain.append((_transform(np.fromstring(o.attrib.get('xyz','0 0 0'),sep=' '), np.fromstring(o.attrib.get('rpy','0 0 0'),sep=' ')), np.fromstring(a.attrib.get('xyz','0 0 1'),sep=' '), j.attrib['name']))
        link=j.find('parent').attrib['link']
    return list(reversed(chain))

def fk(q, chain):
    # Explicit batched loop avoids a dependency on a robot-specific FK package.
    out=np.tile(np.eye(4,dtype=np.float32),(q.shape[0],1,1))
    for ji,(origin,axis,_) in enumerate(chain[:6]):
        out = out @ origin
        a=axis/np.linalg.norm(axis); ang=q[:,ji]; K=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]],dtype=np.float32)
        R=np.eye(3,dtype=np.float32)[None].repeat(len(q),0) + np.sin(ang)[:,None,None]*K + (1-np.cos(ang))[:,None,None]*(K@K)
        jt=np.tile(np.eye(4,dtype=np.float32),(len(q),1,1)); jt[:,:3,:3]=R; out=out@jt
    return out

def sha256(path):
    h=hashlib.sha256();
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def build(data_root, out_root, urdf):
    out_root.mkdir(parents=True, exist_ok=True); (out_root/'episodes').mkdir(exist_ok=True)
    chain=load_chain(urdf); files=sorted((data_root/'data').glob('chunk-*/*.parquet'))
    manifest={'format_version':1,'source':str(data_root.resolve()),'urdf':str(urdf.resolve()),'joint_order':['joint1','joint2','joint3','joint4','joint5','joint6'],'tcp_link':'link6','episodes':{},'source_files':{}}
    for f in tqdm(files,desc='Reading RoboDojo parquet'):
        df=pd.read_parquet(f,columns=['observation.state','episode_index','frame_index'])
        manifest['source_files'][str(f.relative_to(data_root))]=sha256(f)
        for eid,g in df.groupby('episode_index',sort=False):
            q=np.stack(g['observation.state'].to_numpy()).astype(np.float32)
            poses=np.stack([fk(q[:,0:6],chain), fk(q[:,7:13],chain)],axis=1)
            # Robot bases from dual_x5.yml, expressed in one shared world frame.
            # IsaacLab config stores quaternions as wxyz: [0.707, 0, 0, 0.707].
            base_rpy = Rotation.from_quat([0.0, 0.0, 0.707, 0.707]).as_euler('xyz')
            bases=np.array([_transform([-0.3,-0.45,0.765],base_rpy),
                            _transform([0.3,-0.45,0.765],base_rpy)],dtype=np.float32)
            poses=np.einsum('rij,erjkl->erikl',bases,poses[:,None]) if False else poses
            poses=np.einsum('aij,najk->naik',bases,poses)
            ep_path=out_root/'episodes'/f'{int(eid):06d}.npz'; np.savez_compressed(ep_path, pose=poses, source_frame_index=g['frame_index'].to_numpy(np.int64), arm_mask=np.array([True,True]))
            manifest['episodes'][str(int(eid))]={'length':int(len(g)),'file':str(ep_path.relative_to(out_root))}
    (out_root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Wrote {len(manifest["episodes"])} episodes to {out_root}')

def validate(sidecar):
    m=json.loads((sidecar/'manifest.json').read_text()); checked=0
    for eid,meta in tqdm(m['episodes'].items(),desc='Validating geometry'):
        with np.load(sidecar/meta['file'], allow_pickle=False) as z:
            p=z['pose']; r=p[...,:3,:3]
            assert np.isfinite(p).all() and np.max(np.abs(np.matmul(r.transpose(0,1,3,2),r)-np.eye(3)))<2e-4
            assert np.allclose(np.linalg.det(r),1,atol=2e-4); assert np.all(np.diff(z['source_frame_index'])>=0); checked+=len(p)
    print(f'Validated {checked} frames, {len(m["episodes"])} episodes')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,default=DEFAULT_DATA); ap.add_argument('--sidecar',type=Path,required=True); ap.add_argument('--urdf',type=Path,default=DEFAULT_URDF); ap.add_argument('--validate',action='store_true'); a=ap.parse_args()
    if not a.validate: build(a.data_root,a.sidecar,a.urdf)
    validate(a.sidecar)
