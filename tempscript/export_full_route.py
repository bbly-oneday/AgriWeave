"""显式导出完整参考线；日常批处理不额外保存拼接几何。"""
import argparse,sqlite3,sys
from pathlib import Path
import geopandas as gpd
from pyproj import Transformer
from shapely.affinity import translate
from shapely.ops import transform
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import route_planner as io

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--field-id',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    source=io.resolve_reference_gpkg(a.source);out=Path(a.out).resolve()
    if not out.is_relative_to(ROOT/'outputs') or out.exists():raise ValueError('EXPORT_REQUIRES_NEW_OUTPUT_IN_OUTPUTS')
    records=[]
    with sqlite3.connect(f'file:{source}?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row;db.execute('BEGIN')
        crs=db.execute('SELECT definition FROM gpkg_spatial_ref_sys WHERE srs_id=(SELECT srs_id FROM gpkg_contents WHERE table_name="route_segments")').fetchone()[0]
        for fid in a.field_id.split(','):
            detail,ctx,_,_=io.read_reference_field(db,fid)
            converter=None if ctx['metric_crs']=='LOCAL_METRIC' else Transformer.from_crs(ctx['metric_crs'],crs,always_xy=True)
            rows=[{**r,'geometry':io.wkt.loads(r['geometry_wkt'])} for r in detail['full_reference_operations']]
            for r in io._reference_itineraries(rows):
                g=translate(r['geometry'],*ctx['origin']);g=g if converter is None else transform(converter.transform,g)
                records.append(dict(field_id=fid,component=r['component'],reference_complete=bool(detail.get('full_reference_connected')),
                    physical_vehicle_certification='NOT_EVALUATED',geometry=g))
    if not records:raise ValueError('NO_KNOWN_REFERENCE_ROUTE')
    out.parent.mkdir(parents=True,exist_ok=True)
    gpd.GeoDataFrame(records,crs=crs).to_file(out,layer='full_route',driver='GPKG')
    print(out)
