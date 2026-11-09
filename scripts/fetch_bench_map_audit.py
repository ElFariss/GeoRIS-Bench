"""Fetch public map sources for the geometry audit into a private cache."""
import argparse
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import pyogrio
import requests
from pyproj import Transformer

D = None
WCS = "https://service.pdok.nl/rws/actueel-hoogtebestand-nederland/wcs/v1_0/"
BAG_INDEX = "https://data.3dbag.nl/v20250903/tile_index.fgb"
OSM = "https://www.openstreetmap.org/api/0.6/map"

def fetch(file, url, params=None, expected=None):
    p=D/file
    request=requests.Request('GET',url,params=params).prepare()
    if not p.exists():
        for attempt in range(2):
            try:
                with requests.get(request.url,timeout=(12,30),stream=True) as r:
                    r.raise_for_status()
                    with p.with_suffix(p.suffix+'.part').open('wb') as f:
                        chunks = (r.raw.stream(1024*1024, decode_content=False)
                                  if p.suffix == ".gz" else r.iter_content(1024*1024))
                        for b in chunks:
                            f.write(b)
                p.with_suffix(p.suffix+'.part').replace(p)
                break
            except requests.RequestException:
                if attempt: raise
    with p.open('rb') as f:
        h=hashlib.file_digest(f,'sha256').hexdigest()
    if expected and h!=expected: raise RuntimeError(f'Hash mismatch {p}: {h}')
    return {'file':p.name,'url':request.url,'bytes':p.stat().st_size,'sha256':h}

def land(a):
    n=a['id']
    r={'id':n}
    if a['kind']=='netherlands':
        bounds=a['bounds_m']; x0,y0,x1,y1=bounds
        # Use the immutable release index and the publisher's per-tile hashes,
        # not ambiguous live API pagination counts.
        index={'file':'3dbag-v20250903-index.fgb', 'url':BAG_INDEX}
        ip=D/index['file']
        index['bytes']=ip.stat().st_size
        with ip.open('rb') as f:
            index['sha256']=hashlib.file_digest(f,'sha256').hexdigest()
        _,_,_,fields=pyogrio.raw.read(ip,bbox=tuple(bounds),read_geometry=False)
        if not len(fields[0]):
            raise ValueError(f'No 3DBAG tiles intersect {n}')
        tiles=[]
        for tile, sha, url in sorted(zip(*fields,strict=True)):
            if not url or '/v20250903/' not in url or not sha:
                raise ValueError((tile,sha,url))
            tiles.append({**fetch(url.split('/')[-1],url,expected=sha),'tile':tile})
        r['buildings']=tiles
        r['buildings_index']=index
        params={'service':'WCS','version':'2.0.1','request':'GetCoverage','CoverageID':'dtm_05m','crs':'http://www.opengis.net/def/crs/EPSG/0/28992','format':'image/tiff','scalesize':f'x({2*(x1-x0)}),y({2*(y1-y0)})','subset':[f'x({x0},{x1})',f'y({y0},{y1})']}
        r['terrain']=fetch(f'{n}.tif',WCS,params)
    elif a['kind']=='berlin':
        r['buildings']=[fetch(url.split('/')[-1],url) for url in a['buildings_urls']]
        r['terrain']=fetch(a['terrain_url'].split('/')[-1],a['terrain_url'])
    else:
        r['buildings']=[fetch(f'{n}.gml',a['buildings_url'],expected=a['buildings_sha256'])]
        r['terrain']=fetch(f'{n}.tif',a['terrain_url'],expected=a['terrain_sha256'])
    print(n,'land',sum(x['bytes'] for x in r['buildings']),r['terrain']['bytes'],flush=True)
    return r

def osm_all(areas):
    out={}
    for a in areas:
        b=Transformer.from_crs(a['epsg'],4326,always_xy=True).transform_bounds(*a['bounds_m'])
        out[a['id']]=fetch(f'{a["id"]}.osm',OSM,{'bbox':','.join(f'{i:.8f}' for i in b)})
        print(a['id'],'osm',out[a['id']]['bytes'],flush=True)
    return out

def main():
    global D
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--input',type=Path,required=True)
    args=ap.parse_args()
    D=args.input
    D.mkdir(parents=True,exist_ok=True)
    a=json.loads(args.config.read_text())['areas']
    if any(x['kind']=='netherlands' for x in a):
        fetch('3dbag-v20250903-index.fgb',BAG_INDEX)
    with cf.ThreadPoolExecutor(max_workers=4) as p:
        osm=p.submit(osm_all,a)
        lands=list(p.map(land,a))
    osms=osm.result()
    for r in lands:
        r['osm']=osms[r['id']]
    (D/'downloads.json').write_text(json.dumps(lands,indent=2)+'\n')


if __name__=='__main__':
    main()
