"""V7 公共文件工具：原子写入、单一项目配置与旧配置读取适配。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_json(path: Path, content: Any) -> None:
    """将JSON写入同目录临时文件后原子替换目标；拒绝NaN。它保证单次文件替换，不提供多写者锁或数据库事务。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


# 单一用户配置；输出里的参数快照是审计证据，不能反过来当作日常配置入口。
PROJECT_CONFIG = Path(__file__).resolve().parents[1] / 'config.json'
CONFIG_FORMAT = 'V7_PROJECT_CONFIG_1'
LEGACY_CONFIG_SELECTORS = {
    'route_approx.json': 'routes:reference',
    'route_approx_geometry.json': 'routes:geometry',
    'route_approx_operational.json': 'routes:operational',
    'route_separate_headland.json': 'routes:separate_headland',
    'work_time_efficiency_full.json': 'efficiency:continuous',
    'work_time_efficiency_full_v1.json': 'efficiency:constant_full',
    'work_time_efficiency.json': 'efficiency:constant_body',
    'source_layout_20261002.json': 'compatibility:legacy_api',
    'trusted_swath_bundles.json': 'compatibility:trusted_swath_bundles',
}


def config_reference(path):
    """解析 config.json#routes:operational；历史文件名只在已删除时转发。

    外部/封存的旧配置若仍存在，按其原内容读取，不能偷偷替换历史参数。
    路径相对调用者；配置中的数据路径另由 resolve_input_path 相对配置定位。
    """
    raw, separator, selector = str(path or PROJECT_CONFIG).partition('#')
    file = Path(raw).resolve()
    if not separator and not file.exists() and file.parent == PROJECT_CONFIG.parent/'configs':
        selector = LEGACY_CONFIG_SELECTORS.get(file.name, '')
        if selector:file = PROJECT_CONFIG
    return file, selector


def read_json(path):
    """读取真实JSON；带#节选时从统一项目配置提取对应模式。存在的外部旧文件按原内容读取，不偷换历史参数。"""
    file, selector = config_reference(path)
    data = json.loads(file.read_text(encoding='utf-8'))
    if not selector:return data
    section, _, profile = selector.partition(':')
    if not isinstance(data,dict):raise ValueError('CONFIG_MUST_BE_OBJECT')
    if data.get('_format') != CONFIG_FORMAT:
        raise ValueError('CONFIG_SELECTOR_REQUIRES_PROJECT_CONFIG')
    return _config_section(data, section, profile or None)


def _config_section(data, section, profile=None):
    """深拷贝选定模式；效率的速度来自vehicle、停顿来自efficiency.stops，避免重复维护。未知模式抛出ValueError。"""
    import copy
    if section in ('routes','efficiency'):
        group = data[section]
        key = profile or group['active_profile']
        if key not in group['profiles']:raise ValueError(f'UNKNOWN_CONFIG_PROFILE: {section}:{key}')
        selected = copy.deepcopy(group['profiles'][key])
        if section == 'efficiency':
            selected['speeds'] = {name:data['vehicle'][name] for name in
                ('work_speed_mps','turn_speed_mps','transit_speed_mps','reverse_speed_mps')}
            selected['stops'] = copy.deepcopy(group['stops'])
        return selected
    if section == 'compatibility':return copy.deepcopy(data[section][profile])
    raise ValueError('UNKNOWN_CONFIG_SECTION: '+section)


def config_section(path, section, profile=None):
    """统一配置提取有效参数；独立旧配置继续通过原有数值/未知键校验。"""
    file, selector = config_reference(path)
    data = json.loads(file.read_text(encoding='utf-8'))
    if not isinstance(data,dict):raise ValueError('CONFIG_MUST_BE_OBJECT')
    if data.get('_format') != CONFIG_FORMAT:
        if selector:raise ValueError('CONFIG_SELECTOR_REQUIRES_PROJECT_CONFIG')
        return data
    if selector:
        selected_section, _, selected_profile = selector.partition(':')
        if selected_section != section:raise ValueError('CONFIG_SECTION_MISMATCH')
        profile = selected_profile or profile
    return _config_section(data,section,profile)


def scene_config(path):
    """剥离说明、路线、运行和兼容信息后交给冻结场景/条带算法。

    只做配置适配，不修改任何几何或参数值。每田场景不能复制旧API映射等
    项目级信息；封存包保存这份有效配置，以及完整用户配置的独立来源快照。
    """
    data = read_json(path)
    if not isinstance(data,dict):raise ValueError('CONFIG_MUST_BE_OBJECT')
    if data.get('_format') != CONFIG_FORMAT:return data
    return {key:value for key,value in data.items() if key in
        ('vehicle','planning','crs','travel','obstacles','start','end','target','name')}


def project_config(path=PROJECT_CONFIG):
    """检查统一文件结构，未知参数不能因分组/说明而被默默忽略。"""
    data=read_json(config_reference(path)[0])
    if not isinstance(data,dict):raise ValueError('CONFIG_MUST_BE_OBJECT')
    if data.get('_format') != CONFIG_FORMAT:raise ValueError('PROJECT_CONFIG_REQUIRED')
    allowed={'_format','_meta','_help','vehicle','planning','routes','efficiency','execution','input','compatibility'}
    if set(data)!=allowed:raise ValueError('PROJECT_CONFIG_KEYS_MISMATCH: '+str(sorted(set(data)^allowed)))
    for section in allowed-{'_format'}:
        if not isinstance(data[section],dict):raise ValueError('CONFIG_SECTION_MUST_BE_OBJECT: '+section)
    for section,keys in [('execution',{'workers','other_stage_workers','worker_memory_mib','memory_budget_mib','field_timeout','retry_failed','input_chunk_size','plot_fields'}),('input',{'default_gpkg','default_layer'})]:
        if set(data[section])!=keys:raise ValueError('CONFIG_SECTION_KEYS_MISMATCH: '+section)
    for section in ('routes','efficiency'):
        keys={'active_profile','profiles'} | ({'stops'} if section=='efficiency' else set())
        if set(data[section])!=keys:raise ValueError('CONFIG_SECTION_KEYS_MISMATCH: '+section)
        group=data[section]
        if not isinstance(group['profiles'],dict) or not group['profiles']:
            raise ValueError('CONFIG_PROFILES_MUST_BE_OBJECT: '+section)
        if not isinstance(group['active_profile'],str) or group['active_profile'] not in group['profiles']:
            raise ValueError('UNKNOWN_CONFIG_PROFILE: '+section)
        for name, values in group['profiles'].items():
            if not isinstance(values,dict):raise ValueError('CONFIG_PROFILE_MUST_BE_OBJECT: '+section+':'+name)
    if set(data['compatibility'])!={'legacy_api','trusted_swath_bundles'} or any(not isinstance(v,dict) for v in data['compatibility'].values()):
        raise ValueError('INVALID_COMPATIBILITY_CONFIG')
    path=data['input']['default_gpkg'];layer=data['input']['default_layer']
    if not isinstance(path,str) or not path.strip() or (layer is not None and (not isinstance(layer,str) or not layer.strip())):
        raise ValueError('INVALID_INPUT_CONFIG')
    return data


def resolve_input_path(path, config=PROJECT_CONFIG):
    """配置内相对数据路径以配置文件所在目录为基准；绝对路径不改变。这与命令行相对执行目录的路径不同。"""
    file,_=config_reference(config)
    candidate=Path(path)
    return candidate.resolve() if candidate.is_absolute() else (file.parent/candidate).resolve()


def validate_execution(values):
    """拒绝布尔冒充数字、非有限值及越界预算；配置和CLI仍由调度器复核。"""
    import math
    for key in ('workers','other_stage_workers','retry_failed','input_chunk_size'):
        x=values[key]
        if type(x) is not int:raise ValueError('INVALID_EXECUTION_INTEGER: '+key)
    if values['workers']<0 or values['other_stage_workers']<1 or not 0<=values['retry_failed']<=5 or not 1<=values['input_chunk_size']<=2000:
        raise ValueError('INVALID_EXECUTION_RANGE')
    for key in ('worker_memory_mib','memory_budget_mib','field_timeout'):
        x=values[key]
        if type(x) not in (int,float) or not math.isfinite(x) or x<0 or (key!='memory_budget_mib' and x==0):raise ValueError('INVALID_EXECUTION_NUMBER: '+key)
    if type(values['plot_fields']) is not bool:raise ValueError('INVALID_EXECUTION_BOOLEAN: plot_fields')
