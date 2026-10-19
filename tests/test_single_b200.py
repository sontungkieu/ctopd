"""Offline checks for the real-workload launcher; no ML imports or GPU required."""
import ast
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('launcher', ROOT / 'tools/launch_single_b200.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def method_without_ml_imports(relative, name):
    tree = ast.parse((ROOT / relative).read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
    scope = {}
    exec(compile(ast.fix_missing_locations(module), relative, 'exec'), scope)
    return scope[name]


class LauncherTests(unittest.TestCase):
    def plan(self, *arguments):
        return launcher.build_plan(launcher.parser().parse_args(['--run-id', 'test-real', *arguments]))

    def test_real_workload_and_one_gpu_topology(self):
        plan = self.plan()
        flags = plan['flags']
        for key, value in {'train_batch_size': 512, 'rollout_batch_size': 512, 'max_steps': 100,
                           'generate_max_len': 8192, 'prompt_max_len': 2048, 'max_len': 10240,
                           'micro_train_batch_size': 4, 'ctkd_topk': 16, 'eval_math_n_samples': 32,
                           'eval_math_max_tokens': 8192, 'eval_steps': 10, 'save_steps': 10}.items():
            self.assertEqual(flags[key], value, key)
        for key in ['num_nodes', 'num_gpus_per_node', 'teacher_dp_size', 'teacher_tp_size',
                    'teacher_pp_size', 'rollout_num_engines', 'rollout_tp_size']:
            self.assertEqual(flags[key], 1, key)
        self.assertEqual(plan['gradient_accumulation_microsteps'], 128)

    def test_flags_exist_in_actual_argument_dataclasses(self):
        fields = set()
        for path in (ROOT / 'kdflow/arguments').glob('*_args.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    fields.add(node.target.id)
        self.assertFalse(set(self.plan()['flags']) - fields)

    def test_micro_batch_changes_preserve_global_workload_and_baseline(self):
        plan = self.plan('--micro-batch', '1', '--algorithm', 'strict-full')
        self.assertEqual(plan['gradient_accumulation_microsteps'], 512)
        self.assertEqual(plan['flags']['train_batch_size'], 512)
        self.assertEqual(plan['flags']['kd_algorithm'], 'byte_align_ctkd')

    def test_simct_preset_changes_only_algorithm_and_adds_disabled_guard(self):
        baseline = self.plan('--algorithm', 'strict-full')['flags']
        flags = self.plan('--algorithm', 'simct')['flags']
        self.assertEqual(flags['kd_algorithm'], 'simct')
        self.assertEqual(flags['simct_gh_mask_threshold'], 0)
        self.assertEqual({k: v for k, v in flags.items() if k not in {'kd_algorithm', 'simct_gh_mask_threshold'}},
                         {k: v for k, v in baseline.items() if k != 'kd_algorithm'})

    def test_mp_mode_presets_preserve_full_workload(self):
        baseline = self.plan()['flags']
        variants = ['mp-atomic', *['mp-fixed' + str(n) for n in range(1, 6)],
                    'mp-random1-2', 'mp-random2-5', 'mp-grass-chunk', 'mp-align']
        for name in variants:
            plan = self.plan('--algorithm', name)
            self.assertTrue(plan['flags']['kd_algorithm'].startswith('mp_opd_'))
            for key, value in baseline.items():
                if key not in {'kd_algorithm'}:
                    self.assertEqual(plan['flags'][key], value, (name, key))
            self.assertIn('no B200/full-training qualification', plan['qualification'])
        for n in range(1, 6):
            flags = self.plan('--algorithm', 'mp-fixed' + str(n))['flags']
            self.assertEqual(flags['kd_algorithm'], 'mp_opd_fixed')
            self.assertEqual(flags['mp_opd_fixed_span_length'], n)
        for name, interval in [('mp-random1-2', (1, 2)), ('mp-random2-5', (2, 5))]:
            flags = self.plan('--algorithm', name)['flags']
            self.assertEqual((flags['mp_opd_random_min_length'], flags['mp_opd_random_max_length']), interval)

    def test_mp_flags_exist_in_argument_dataclasses(self):
        fields = set()
        for path in (ROOT / 'kdflow/arguments').glob('*_args.py'):
            fields |= {node.target.id for node in ast.walk(ast.parse(path.read_text()))
                       if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
        for name in ['mp-atomic', 'mp-fixed5', 'mp-random2-5', 'mp-grass-chunk', 'mp-align', 'mp-align-new']:
            self.assertFalse(set(self.plan('--algorithm', name)['flags']) - fields)

    def test_align_new_preset_and_exact_head_guard(self):
        flags = self.plan('--algorithm', 'mp-align-new')['flags']
        self.assertEqual(flags['kd_algorithm'], 'mp_opd_align_new')
        self.assertEqual(flags['mp_opd_align_new_min_descent_fraction'], .1)
        self.assertEqual(flags['mp_opd_align_new_eps_g'], 1e-12)
        self.assertEqual(flags['mp_opd_chunk_source'], 'xtoken')
        for key in ('train_batch_size','rollout_batch_size','max_steps','generate_max_len','eval_steps','eval_math_n_samples'):
            self.assertEqual(flags[key], self.plan()['flags'][key])
        with self.assertRaisesRegex(ValueError, 'exact_head'):
            self.plan('--algorithm','mp-align-new','--mp-geometry','diag')

    def test_dry_run_creates_no_files(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = pathlib.Path(folder) / 'absent'
            raw = subprocess.check_output([sys.executable, str(ROOT / 'tools/launch_single_b200.py'),
                '--root', str(destination), '--dry-run'], text=True)
            self.assertEqual(json.loads(raw)['flags']['max_steps'], 100)
            self.assertFalse(destination.exists())

    def test_existing_output_and_path_traversal_are_refused(self):
        with self.assertRaises(ValueError):
            self.plan('--run-id', '../other-run')
        with tempfile.TemporaryDirectory() as folder:
            plan = self.plan('--root', folder, '--runtime', str(pathlib.Path(folder) / 'runtime'))
            for key in ['student_name_or_path', 'teacher_name_or_path']:
                model = pathlib.Path(plan['flags'][key])
                model.mkdir(parents=True)
                for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors']:
                    (model / name).write_text('{}')
            for path in [pathlib.Path(plan['runtime']) / 'python-b200.sh',
                         pathlib.Path(plan['flags']['train_dataset_path']),
                         pathlib.Path(plan['flags']['eval_math_dataset_path'])]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            pathlib.Path(plan['output']).mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                launcher.validate_files(plan)

    def test_actual_token_normalization_across_world_and_micro_batch_sizes(self):
        prepare = method_without_ml_imports('kdflow/trainer/on_policy_kd_trainer.py', '_prepare_global_batches')
        backward = method_without_ml_imports('kdflow/backend/fsdp/fsdp_strategy.py', 'backward')
        # Unequal sequence lengths make an incorrect mean-of-means observable.
        lengths = [1 + (index * 13 % 97) for index in range(512)]
        gradients = [((index % 7) - 3) * length for index, length in enumerate(lengths)]
        reference = sum(gradients) / sum(lengths)

        class Mask:
            def __init__(self, count): self.count = count
            def sum(self): return self.count

        class Loss:
            def __init__(self, value, sink): self.value, self.sink = value, sink
            def __truediv__(self, divisor): return Loss(self.value / divisor, self.sink)
            def backward(self): self.sink.append(self.value)

        for world, micro in [(8, 4), (1, 4), (1, 1)]:
            batches = [{'stu_loss_mask': Mask(sum(lengths[i:i + micro])),
                        'gradient': sum(gradients[i:i + micro])} for i in range(0, 512, micro)]
            trainer = types.SimpleNamespace(args=types.SimpleNamespace(train=types.SimpleNamespace(use_dynamic_bsz=False)))
            batches = prepare(trainer, batches, 512 // micro)[0]
            rank_gradients = []
            for rank in range(world):
                strategy = types.SimpleNamespace(step=0, accumulated_gradient=512 // micro // world)
                sink = []
                for batch in batches[rank::world]:
                    loss = Loss(batch['gradient'] / batch['avg_micro_batch_token_num'], sink)
                    backward(strategy, loss, None, None)
                self.assertEqual(strategy.step, 0)
                rank_gradients.append(sum(sink))
            self.assertAlmostEqual(sum(rank_gradients) / world, reference, places=12)


if __name__ == '__main__':
    unittest.main()
