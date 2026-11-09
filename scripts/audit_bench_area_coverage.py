"""Phase 0b geometry-only building and ground coverage; writes aggregate JSON."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

from lxml import etree as ET
import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from shapely import make_valid
from shapely.geometry import LineString, Polygon, box
from shapely.ops import polygonize, unary_union


def date_counts(counter, value):
    if value:
        counter[str(value)[:10]] += 1


def all_polys(tree):
    """Flatten CityJSON boundary nesting, preserving polygon rings."""
    if not tree:
        return
    if isinstance(tree[0], list) and tree[0] and isinstance(tree[0][0], int):
        yield tree
        return
    for node in tree:
        if isinstance(node, list):
            yield from all_polys(node)


def shape(rings):
    p = make_valid(Polygon(rings[0], rings[1:]))
    if p.area > 0:
        return p
    return None


def inside(p, region):
    x0, y0, x1, y1 = region.bounds
    q = p.representative_point()
    return x0 <= q.x < x1 and y0 <= q.y < y1


def cityjson(files, region):
    models, ids = [], set()
    stats = Counter()
    dates, campaigns, quality, no_lod = Counter(), Counter(), Counter(), Counter()
    for file in files:
        with gzip.open(file,'rt') if file.suffix=='.gz' else file.open() as f:
            d = json.load(f)
        meta = d.get('metadata',{})
        scale, shift = [np.asarray(d['transform'][k]) for k in ('scale','translate')]
        crs = meta['referenceSystem']
        if not crs.endswith('/7415'):
            raise ValueError(crs)
        v = np.asarray(d['vertices']) * scale + shift
        objects = d['CityObjects']
        for key,parent in objects.items():
            if parent['type'] != 'Building':
                continue
            if key in ids:
                raise ValueError(f'duplicate BAG ID {key}')
            ids.add(key)
            attrs = parent.get('attributes',{})
            g0 = [p for g in parent.get('geometry',[]) if g['lod'] == '0'
                  for rings in all_polys(g['boundaries']) if (p := shape([v[r,:2] for r in rings])) is not None]
            if not g0:
                stats['missing_source_footprint'] += 1
                continue
            footprint = unary_union(g0)
            if not footprint.intersects(region):
                continue
            selected = inside(footprint, region)
            if selected:
                stats['provider_buildings'] += 1
                date_counts(dates, attrs.get('b3_pw_datum'))
                date_counts(campaigns, attrs.get('b3_pw_bron'))
                quality[str(attrs.get('b3_kwaliteitsindicator'))] += 1
            geoms = [g for k in [key,*parent.get('children',[])]
                     for g in objects[k].get('geometry',[]) if g['lod'] == '2.2']
            surfaces = {}
            for g in geoms:
                spec = g.get('semantics',{})
                def flatten(values):
                    for x in values:
                        if isinstance(x,list):
                            yield from flatten(x)
                        else:
                            yield x
                for rings,i in zip(all_polys(g['boundaries']), flatten(spec.get('values',[])), strict=True):
                    if i is not None:
                        k=spec['surfaces'][i]['type']
                        surfaces.setdefault(k,[]).extend(v[r] for r in rings)
            if not {'WallSurface','RoofSurface','GroundSurface'} <= surfaces.keys():
                if selected:
                    no_lod['missing_lod22_or_semantics'] += 1
                continue
            xyz = np.concatenate([p for group in surfaces.values() for p in group])
            roof_z = max(p[:,2].max() for p in surfaces['RoofSurface'])
            ground_z = min(p[:,2].min() for p in surfaces['GroundSurface'])
            if not np.isfinite(xyz).all() or roof_z<=ground_z:
                if selected:
                    no_lod['nonfinite_or_nonpositive'] += 1
                continue
            models.append(footprint)
            if selected:
                stats['usable_lod2'] += 1
    return models, dict(stats), {'point_cloud_years': dict(dates), 'point_cloud_sources': dict(campaigns),
                                 'provider_quality_indicator': dict(quality), 'rejections': dict(no_lod)}


def gml_ring(ring):
    p = ring.find('./{*}posList')
    if p is not None:
        coords = np.fromstring(p.text or '', sep=' ')
    else:
        coords = np.fromstring(' '.join(x.text for x in ring.findall('./{*}pos')), sep=' ')
    return coords.reshape(-1,3)


def gml_surface(surface):
    for poly in surface.findall('.//{*}Polygon'):
        outer = poly.find('./{*}exterior/{*}LinearRing')
        if outer is None:
            continue
        rings = [gml_ring(outer)]
        rings.extend(gml_ring(x) for x in poly.findall('./{*}interior/{*}LinearRing'))
        if len(rings[0]) >= 4:
            yield rings


def gml_models(files, region):
    models, ids = [], set()
    stats, dates, no_lod = Counter(), Counter(), Counter()
    for file in files:
        for _, member in ET.iterparse(str(file), events=('end',),tag='{*}cityObjectMember',
                                      resolve_entities=False, no_network=True):
            b = member.find('./{*}Building')
            if b is not None:
                ground = [r for s in b.findall('.//{*}GroundSurface') for r in gml_surface(s)]
                ground_poly = [p for r in ground if (p := shape([x[:,:2] for x in r])) is not None]
                if ground_poly:
                    footprint = unary_union(ground_poly)
                    ident = b.get('{http://www.opengis.net/gml}id')
                    if footprint.intersects(region) and ident not in ids:
                        if ident:
                            ids.add(ident)
                        selected = inside(footprint, region)
                        if selected:
                            stats['provider_buildings_with_ground_outline'] += 1
                            date_counts(dates,b.findtext('./{*}creationDate'))
                        roof = [r for s in b.findall('.//{*}RoofSurface') for r in gml_surface(s)]
                        wall = [r for s in b.findall('.//{*}WallSurface') for r in gml_surface(s)]
                        if not roof or not wall:
                            if selected:
                                no_lod['missing_roof_or_wall'] += 1
                        else:
                            xyz = np.concatenate([r[0] for r in roof+wall+ground])
                            height = max(r[0][:,2].max() for r in roof) - min(r[0][:,2].min() for r in ground)
                            if not np.isfinite(xyz).all() or height <= 0:
                                if selected:
                                    no_lod['nonfinite_or_nonpositive'] += 1
                            else:
                                models.append(footprint)
                                if selected:
                                    stats['usable_lod2'] += 1
                else:
                    stats['whole_tiles_buildings_without_ground_outline'] += 1
            member.clear()
            while member.getprevious() is not None:
                del member.getparent()[0]
    return models, dict(stats), {'creation_dates': dict(dates), 'rejections': dict(no_lod)}


def extract_gml(archives, directory):
    files = []
    for archive in archives:
        if archive.suffix != '.zip':
            files.append(archive)
            continue
        with ZipFile(archive) as z:
            names = [n for n in z.namelist() if n.lower().endswith(('.gml','.xml'))]
            if len(names) != 1:
                raise ValueError(f'expected one CityGML file in {archive}: {names}')
            target = directory / f'{archive.stem}.gml'
            target.write_bytes(z.read(names[0]))
            files.append(target)
    return files


def get_height(tags):
    for k in ('height','building:height'):
        t = tags.get(k,'').strip().lower()
        m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*(m|ft|feet)?', t)
        if m:
            h = float(m.group(1)) * (0.3048 if m.group(2) in {'ft','feet'} else 1)
            if 0 < h <= 300:
                return 'explicit_height'
    try:
        h = 3 * float(tags.get('building:levels','nan'))
    except ValueError:
        h = 0
    return 'levels' if 0 < h <= 300 else 'default'


def osm_buildings(file, crs, region):
    tree = ET.parse(str(file), parser=ET.XMLParser(resolve_entities=False,no_network=True))
    tr = Transformer.from_crs(4326,crs,always_xy=True)
    nodes = {e.get('id'):tr.transform(float(e.get('lon')),float(e.get('lat'))) for e in tree.findall('node')}
    ways = {e.get('id'):e for e in tree.findall('way')}

    def tags(el):
        return {t.get('k'):t.get('v') for t in el.findall('tag')}

    def coords(el):
        refs = [n.get('ref') for n in el.findall('nd')]
        return [nodes[r] for r in refs] if all(r in nodes for r in refs) else []

    entries = []
    used = set()
    diagnostics = Counter()
    for relation in tree.findall('relation'):
        t = tags(relation)
        if t.get('building') in {None,'no'} or t.get('type') != 'multipolygon':
            continue
        members = {role: [m.get('ref') for m in relation.findall('member')
                          if m.get('type')=='way' and (m.get('role') or 'outer')==role]
                   for role in ('outer','inner')}
        polys = {}
        complete = True
        for role, refs in members.items():
            lines = [LineString(c) for k in refs if k in ways and len(c := coords(ways[k])) >= 2]
            polys[role] = list(polygonize(lines))
            if len(lines) != len(refs):
                complete = False
        if not complete or not polys['outer']:
            diagnostics['incomplete_building_multipolygon'] += 1
            continue
        p = unary_union(polys['outer']).difference(unary_union(polys['inner']))
        used.update(members['outer']+members['inner'])
        if p.area > 0 and inside(p,region):
            entries.append((p,get_height(t),t.get("location") == "underground" or t.get("underground") == "yes"))
    for key,e in ways.items():
        t = tags(e)
        if key in used or t.get('building') in {None,'no'}:
            continue
        c = coords(e)
        if len(c) < 4 or c[0] != c[-1]:
            diagnostics['incomplete_or_open_building_way'] += 1
            continue
        p = shape([c])
        if p is not None and inside(p,region):
            entries.append((p,get_height(t),t.get("location") == "underground" or t.get("underground") == "yes"))
    return entries, dict(diagnostics)


def terrain(file, region, building_union):
    outside = {}
    if file.suffix == '.zip':
        with ZipFile(file) as z:
            names = [n for n in z.namelist() if n.lower().endswith(('.csv','.txt','.xyz'))]
            if len(names) != 1:
                raise ValueError((file,names))
            with z.open(names[0]) as f:
                first = f.readline().decode('utf-8-sig')
                delim = ';' if ';' in first else ',' if ',' in first else None
                skip = int(any(k.isalpha() for k in first))
            with z.open(names[0]) as f:
                xyz = np.loadtxt(f,delimiter=delim,skiprows=skip,usecols=(0,1,2))
        x0,y0,x1,y1 = region.bounds
        local = xyz[(xyz[:,0]>=x0)&(xyz[:,0]<x1)&(xyz[:,1]>=y0)&(xyz[:,1]<y1)]
        values = local[:,2]
        dx,dy = (float(np.median(np.diff(np.unique(local[:,k])))) for k in (0,1))
        count = round((x1-x0)*(y1-y0)/(dx*dy))
        if len(np.unique(local[:,:2],axis=0)) != len(local):
            raise ValueError('duplicate DGM grid coordinates')
        valid = values[np.isfinite(values)&(values>-1000)&(values<10000)]
    else:
        with rasterio.open(file) as r:
            dx,dy = r.res
            w = from_bounds(*region.bounds,transform=r.transform).round_offsets().round_lengths()
            grid = r.read(1,window=w,boundless=True,masked=True)
            tf = r.window_transform(w)
        count = grid.size
        numeric = np.asarray(grid.data)
        good = ~np.ma.getmaskarray(grid)&np.isfinite(numeric)&(numeric>-1000)&(numeric<10000)
        valid = numeric[good]
        occupied = rasterize([building_union],out_shape=grid.shape,transform=tf).astype(bool)
        outside = {'outside_lod2_footprint_cells': int((~occupied).sum()),
                   'valid_outside_lod2_footprint_cells': int((good&~occupied).sum()),
                   'nodata_inside_lod2_footprint_cells': int((~good&occupied).sum())}
    if len(valid)==0:
        raise ValueError(f'empty DTM {file}')
    p05,p95 = np.percentile(valid,[5,95])
    return {'resolution_m':[dx,dy],'expected_cells':count,'valid_cells':int(len(valid)),
            'nodata_fraction':float(1-len(valid)/count),'min_m':float(valid.min()),'max_m':float(valid.max()),
            'p05_m':float(p05),'p95_m':float(p95),'p05_p95_relief_m':float(p95-p05), **outside}


def analyze(a, manifest, data):
    region = box(*a['bounds_m'])
    files = [data/x['file'] for x in manifest['buildings']]
    if a['kind']=='netherlands':
        models,bcounts,details = cityjson(files,region)
    else:
        models,bcounts,details = gml_models(extract_gml(files,data),region)
    if not models:
        raise ValueError(f'{a["id"]}: empty LoD2')
    combined = unary_union(models)
    osm, osm_diagnostics = osm_buildings(data/manifest['osm']['file'],a['epsg'],region)
    tallies,baseline = Counter(),Counter()
    weighted_total, weighted_overlap, biggest_unmatched = 0.,0.,0.
    surface_total, surface_overlap = 0.,0.
    underground, surface_match = 0,0
    for p,tag,below_ground in osm:
        baseline[tag] += 1
        area = p.area
        overlap = combined.intersection(p).area
        weighted_total += area
        weighted_overlap += overlap
        if below_ground:
            underground += 1
        else:
            surface_total += area
            surface_overlap += overlap
            surface_match += int(overlap >= area*.5)
        if overlap >= area*.5:
            tallies['lod2_match'] += 1
        elif tag == 'default':
            biggest_unmatched = max(biggest_unmatched, area)
            tallies['default_fallback'] += 1
        else:
            biggest_unmatched = max(biggest_unmatched, area)
            tallies['osm_tag_fallback'] += 1
    assert sum(tallies.values())==len(osm)
    return {'id':a['id'],'role':a['role'],'provider_model_counts':bcounts,'provider_details':details,
            'osm_footprints':len(osm),'osm_footprint_against_lod2':dict(tallies),
            'osm_lod1_baseline':dict(baseline),'osm_diagnostics':osm_diagnostics,
            'osm_total_footprint_area_overlap_fraction': weighted_overlap/weighted_total if weighted_total else None,
            'osm_largest_unmatched_footprint_m2': biggest_unmatched,
            'osm_surface_only_sensitivity': {'explicitly_underground_excluded':underground,
                'surface_candidate_count':len(osm)-underground,'surface_match_count':surface_match,
                'footprint_area_overlap_fraction':surface_overlap/surface_total if surface_total else None},
            'terrain':terrain(data/manifest['terrain']['file'],region,combined),'sources':manifest}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args = ap.parse_args()
    conf = json.loads(args.config.read_text())
    manifest = {r['id']:r for r in json.loads((args.input/'downloads.json').read_text())}
    for d in manifest.values():
        for row in d['buildings']+[d['terrain'],d['osm']]+([d['buildings_index']] if 'buildings_index' in d else []):
            with (args.input/row['file']).open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=row['sha256']:
                    raise ValueError(f'input changed: {row["file"]}')
    results = []
    for a in conf['areas']:
        res = analyze(a,manifest[a['id']],args.input)
        results.append(res)
        print(a['id'],res['provider_model_counts'],res['osm_footprint_against_lod2'],
              res['terrain']['p05_p95_relief_m'],flush=True)
    args.out.write_text(json.dumps({'scope':'geometry-only; no held-out channels',
                                    'areas':results},indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
