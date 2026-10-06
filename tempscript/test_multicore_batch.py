"""多核调度的资源边界与原生线程隔离；服务器配额用可控夹具验证。"""
import contextlib,io,json,os,pickle,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
TEST_OUTPUT=Path(os.environ.get('V7_TEST_OUTPUT_ROOT',ROOT/'outputs/multicore_batch_tests'))
sys.path.insert(0,str(ROOT/'src'))
import route_planner as m
import main

def scheduler_probe_worker(job,settings,cfg,path,plot):
    # 只验证超过 12 个真实 spawn 子进程和错误隔离，绝不冒充路线计算。
    time.sleep(1)
    result=dict(ok=False,error='SCHEDULER_PROBE_'+str(os.getpid())+'_'+os.environ.get('OMP_NUM_THREADS','UNKNOWN'),elapsed_seconds=1.)
    temp=path+'.tmp'
    with open(temp,'wb') as f:pickle.dump(result,f)
    os.replace(temp,path)


class MulticoreTests(unittest.TestCase):
    def plan(self,workers=0,cpus=128,ram=262144,fields=350,**kw):
        with patch.object(m,'batch_resource_limits',return_value=dict(visible_logical_cpus=cpus,
                usable_cpus=cpus,available_memory_mib=ram,memory_source='fixture')):
            return m.reference_worker_plan(workers,field_count=fields,**kw)
    def test_idle_sample_counts_only_affinity_cores(self):
        with patch('psutil.cpu_percent',return_value=[100,0,100,50]) as sample:
            result=m._cpu_idle_capacity({1,3})
        self.assertEqual(result['cpu_idle_slots'],1)
        self.assertEqual(result['cpu_utilization_pct'],25)
        sample.assert_called_once_with(interval=.25,percpu=True)
    def test_idle_sample_all_idle_and_fully_busy(self):
        for values,expected in (([0]*16,16),([100]*16,1),([25]*16,12)):
            with self.subTest(values=values),patch('psutil.cpu_percent',return_value=values):
                self.assertEqual(m._cpu_idle_capacity()['cpu_idle_slots'],expected)
    def test_invalid_or_missing_cpu_samples_are_unknown(self):
        for values in ([],[float('nan')],[101],[-1],['bad'],[True]):
            with self.subTest(values=values),patch('psutil.cpu_percent',return_value=values):
                self.assertIsNone(m._cpu_idle_capacity())
        with patch('psutil.cpu_percent',return_value=[0]):
            self.assertIsNone(m._cpu_idle_capacity({9}))
        with patch('psutil.cpu_percent',side_effect=PermissionError):
            self.assertIsNone(m._cpu_idle_capacity())
    def test_worker_count_takes_idle_cpu_and_memory_minimum(self):
        for idle,ram,expected in ((4,262144,4),(100,8192,7),(100,262144,12)):
            with self.subTest(idle=idle,ram=ram),patch.object(m,'batch_resource_limits',return_value=dict(
                    usable_cpus=128,available_memory_mib=ram,cpu_idle_slots=idle)):
                self.assertEqual(m.reference_worker_plan(12,field_count=350)['effective_workers'],expected)
    def test_unknown_memory_positive_request_is_also_conservative(self):
        self.assertEqual(self.plan(12,ram=None)['effective_workers'],1)
    def test_default_config_requests_twelve(self):
        cfg=json.loads((ROOT/'config.json').read_text())
        self.assertEqual(cfg['execution']['workers'],12)
        self.assertEqual(cfg['execution']['other_stage_workers'],12)
        self.assertIn('启动时',cfg['_help']['execution.workers'])

    def test_swath_main_default_is_resource_capped(self):
        import swath_batch
        TEST_OUTPUT.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as temp:
            out=Path(temp)/'swaths'
            argv=['main.py','--stage','swaths','--input',str(ROOT.parent/'data/fields2cover_regular_size_5samples.gpkg'),'--out',str(out)]
            def fake_run(*args,**kw):out.mkdir();return 0
            limits=dict(usable_cpus=32,available_memory_mib=4096,cpu_idle_slots=24)
            with patch.object(sys,'argv',argv),patch.object(m,'batch_resource_limits',return_value=limits),patch.object(swath_batch,'run_batch',side_effect=fake_run) as run,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main.main(),0)
            self.assertEqual(run.call_args.kwargs['workers'],3)
            plan=json.loads((out/'worker_plan.json').read_text())
            self.assertEqual(plan['requested_workers'],12)
    def test_swath_main_auto_and_existing_output_protection(self):
        import swath_batch
        TEST_OUTPUT.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as temp:
            out=Path(temp)/'swaths'
            argv=['main.py','--stage','swaths','--workers','0','--input',str(ROOT.parent/'data/fields2cover_regular_size_5samples.gpkg'),'--out',str(out)]
            def fake_run(*args,**kw):out.mkdir();return 0
            limits=dict(usable_cpus=32,available_memory_mib=262144,cpu_idle_slots=8)
            with patch.object(sys,'argv',argv),patch.object(m,'batch_resource_limits',return_value=limits),patch.object(swath_batch,'run_batch',side_effect=fake_run) as run,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main.main(),0)
                self.assertEqual(run.call_args.kwargs['workers'],5)
                before=(out/'worker_plan.json').read_bytes()
                with contextlib.redirect_stderr(io.StringIO()):self.assertEqual(main.main(),2)
                self.assertEqual(run.call_count,1)
                self.assertEqual((out/'worker_plan.json').read_bytes(),before)

    def test_server_auto_uses_more_than_twelve(self):
        self.assertEqual(self.plan()['effective_workers'],128)
    def test_explicit_sixtyfour_workers_supported(self):
        self.assertEqual(self.plan(64)['effective_workers'],64)
    def test_cpu_limit_prevents_oversubscription(self):
        self.assertEqual(self.plan(256,cpus=32)['effective_workers'],32)
    def test_memory_limit_reduces_parallelism(self):
        self.assertEqual(self.plan(cpus=128,ram=8192)['effective_workers'],7)
    def test_small_batch_does_not_spawn_unused_cores(self):
        self.assertEqual(self.plan(fields=5)['effective_workers'],5)
    def test_no_tasks_requires_no_workers(self):
        self.assertEqual(self.plan(fields=0)['effective_workers'],0)
    def test_unknown_memory_auto_falls_back_to_one(self):
        self.assertEqual(self.plan(ram=None)['effective_workers'],1)
    def test_unknown_memory_can_use_explicit_budget(self):
        self.assertEqual(self.plan(ram=None,memory_budget_mib=65536)['effective_workers'],84)
    def test_explicit_budget_cannot_exceed_detected_memory(self):
        self.assertEqual(self.plan(ram=8192,memory_budget_mib=65536)['effective_workers'],10)
    def test_not_enough_memory_is_not_silently_overcommitted(self):
        with self.assertRaisesRegex(ValueError,'INSUFFICIENT_MEMORY'):self.plan(ram=512)
    def test_bad_worker_values(self):
        for value in (True,-1,1.5,'auto'):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'INVALID_WORKERS'):self.plan(value)
    def test_bad_memory_values(self):
        for kw in (dict(worker_memory_mib=0),dict(worker_memory_mib=float('nan')),
                dict(memory_budget_mib=-1),dict(memory_budget_mib=True),dict(memory_budget_mib=float('inf'))):
            with self.subTest(kw=kw),self.assertRaises(ValueError):self.plan(**kw)
    def test_affinity_is_respected(self):
        with patch.object(m.os,'cpu_count',return_value=128),patch.object(m.os,'sched_getaffinity',return_value=set(range(6)),create=True),patch.object(m,'_cgroup_resource_directories',return_value=[]):
            self.assertEqual(m.batch_resource_limits()['usable_cpus'],6)
    def cgroup_discovery_fixture(self,root,line):
        # 真正执行分组/祖先路径发现，仅映射 Linux 两个只读文件系统入口。
        actual=Path;proc=root/'proc_cgroup';proc.write_text(line)
        def path_factory(value):
            if value=='/sys/fs/cgroup':return root
            if value=='/proc/self/cgroup':return proc
            return actual(value)
        with patch.object(m.sys,'platform','linux'),patch.object(m,'Path',side_effect=path_factory):
            return m._cgroup_resource_directories()
    def test_linux_v2_discovers_current_group_and_all_ancestors(self):
        base=TEST_OUTPUT/'tests';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            root=Path(tmp);leaf=root/'department/worker';leaf.mkdir(parents=True)
            found=self.cgroup_discovery_fixture(root,'0::/department/worker\n')
            for path in (leaf,leaf.parent,root):self.assertIn(('unified',path),found)
    def test_linux_v1_discovers_combined_cpu_and_memory_mounts(self):
        base=TEST_OUTPUT/'tests';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            root=Path(tmp);(root/'cpu,cpuacct/team').mkdir(parents=True);(root/'memory/team').mkdir(parents=True)
            found=self.cgroup_discovery_fixture(root,'7:cpu,cpuacct:/team\n5:memory:/team\n')
            self.assertIn(('cpu',root/'cpu,cpuacct/team'),found)
            self.assertIn(('memory',root/'memory/team'),found)
    def test_linux_namespace_parent_path_does_not_escape_mount(self):
        base=TEST_OUTPUT/'tests';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            root=Path(tmp);found=self.cgroup_discovery_fixture(root,'0::/../../host/group\n')
            self.assertTrue(all(path.is_relative_to(root) for _,path in found))

    def test_cgroup_v2_and_ancestor_limits(self):
        base=TEST_OUTPUT/'tests';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            root=Path(tmp);child=root/'child';child.mkdir()
            (root/'cpu.max').write_text('250000 100000');(child/'cpu.max').write_text('800000 100000')
            (root/'memory.max').write_text(str(4*2**30));(root/'memory.current').write_text(str(3*2**30))
            (child/'memory.max').write_text('max');(child/'memory.current').write_text('0')
            with patch.object(m.os,'cpu_count',return_value=128),patch.object(m.os,'sched_getaffinity',return_value=set(range(128)),create=True),patch.object(m,'_cgroup_resource_directories',return_value=[('unified',root),('unified',child)]):
                resources=m.batch_resource_limits()
            self.assertEqual(resources['usable_cpus'],2);self.assertEqual(resources['cgroup_cpu_quota'],2.5)
            self.assertLessEqual(resources['available_memory_mib'],1024)
    def test_cgroup_v1_limits_and_negative_unlimited(self):
        base=TEST_OUTPUT/'tests';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            root=Path(tmp);(root/'cpu.cfs_quota_us').write_text('300000');(root/'cpu.cfs_period_us').write_text('100000')
            (root/'memory.limit_in_bytes').write_text(str(2*2**30));(root/'memory.usage_in_bytes').write_text(str(2**30))
            with patch.object(m.os,'cpu_count',return_value=128),patch.object(m.os,'sched_getaffinity',return_value=set(range(128)),create=True),patch.object(m,'_cgroup_resource_directories',return_value=[('cpu',root),('memory',root)]):
                self.assertEqual(m.batch_resource_limits()['usable_cpus'],3)
                (root/'cpu.cfs_quota_us').write_text('-1')
                self.assertEqual(m.batch_resource_limits()['usable_cpus'],128)
    def test_spawn_receives_one_thread_and_restores_parent_environment(self):
        seen={}
        class Process:
            def start(self):seen.update({key:os.environ.get(key) for key in m._NATIVE_THREAD_ENV})
        with patch.dict(os.environ,{'OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'9'}):
            before={key:os.environ.get(key) for key in m._NATIVE_THREAD_ENV}
            m._start_reference_process(Process())
            self.assertEqual(set(seen.values()),{'1'})
            self.assertEqual(before,{key:os.environ.get(key) for key in m._NATIVE_THREAD_ENV})
    def test_failed_spawn_also_restores_parent_environment(self):
        class Process:
            def start(self):raise OSError('process limit')
        before={key:os.environ.get(key) for key in m._NATIVE_THREAD_ENV}
        with self.assertRaises(OSError):m._start_reference_process(Process())
        self.assertEqual(before,{key:os.environ.get(key) for key in m._NATIVE_THREAD_ENV})
    def test_cli_help_and_auto_default(self):
        p=main.parser();self.assertIn('--worker-memory-mib',p.format_help())
        self.assertIsNone(p.parse_args([]).workers);self.assertEqual(p.parse_args(['--workers','0']).workers,0)
    @unittest.skipUnless(os.environ.get("V7_TEST_SWATH_BUNDLE"), "可选原生spawn测试需要至少20田的已封存包")
    def test_real_spawn_scheduler_accepts_sixteen_slots_with_simulated_server_resources(self):
        # 本机只有 12 核：模拟资源探测，而 spawn/事务/错误隔离全部是真实执行。
        # 这里不运行路线求解，不能声称完成 16 核服务器性能测试。
        base=TEST_OUTPUT/'tests';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp,patch.object(m,'batch_resource_limits',return_value=dict(
                visible_logical_cpus=64,usable_cpus=64,available_memory_mib=131072,memory_source='SIMULATED_SERVER_FIXTURE')):
            with contextlib.redirect_stdout(io.StringIO()):
                bundle=Path(os.environ['V7_TEST_SWATH_BUNDLE'])
                manifest=json.loads((bundle/'prepared/manifest.json').read_text())
                selected=','.join(e['field_id'] for e in manifest['fields'][:20])
                self.assertEqual(len(manifest['fields'][:20]),20)
                summary=m.run_incremental_reference_batch(bundle,
                    Path(tmp)/'sixteen',workers=16,field_id=selected,route_config=ROOT/'config.json#routes:reference',
                    worker_target=scheduler_probe_worker,retries=0,field_timeout=30)
            self.assertEqual(summary['workers'],16);self.assertEqual(summary['peak_active_workers'],16)
            self.assertEqual(summary['state_counts'],{'FAILED':20})
            self.assertFalse(summary['reference_acceptance_passed'])
            import sqlite3
            with sqlite3.connect(Path(tmp)/'sixteen/reference_routes.gpkg') as db:
                self.assertEqual(db.execute('PRAGMA integrity_check').fetchone()[0],'ok')
                self.assertEqual(db.execute("SELECT COUNT(*) FROM field_results WHERE error LIKE 'SCHEDULER_PROBE_%_1'").fetchone()[0],20)
    def test_no_work_resume_does_not_require_new_memory(self):
        self.assertEqual(self.plan(fields=0,ram=0)['effective_workers'],0)

    def test_small_and_invalid_cgroup_formats_do_not_erase_cpu_count(self):
        with patch.object(m,'_cgroup_resource_directories',return_value=[('unified',Path('/missing'))]),patch.object(m.os,'cpu_count',return_value=None),patch.object(m.os,'sched_getaffinity',side_effect=OSError,create=True):
            self.assertEqual(m.batch_resource_limits()['usable_cpus'],1)

if __name__=='__main__':unittest.main(verbosity=2)
