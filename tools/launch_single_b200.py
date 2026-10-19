#!/usr/bin/env python3
"""Launch the real Qwen -> Llama training configuration on one allocated B200.

Standard library only. --dry-run prints a plan without importing the runtime,
checking GPUs, writing files or starting training. There is no reduced workload.
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import shlex
import socket
import subprocess

SOURCE = pathlib.Path(__file__).resolve().parents[1]
ARCHIVE_SHA = '084c321c0b7af1cd3a36d12a595c623567d4a37525aa0eecbc3933aa7f9b15c3'
H1B_ARMS={'h1b-a':'a','h1b-b':'b','h1b-b-bl':'b_bl','h1b-c-str':'c_str','h1b-c':'c'}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=pathlib.Path,
                   default=pathlib.Path('/workspace/storage-shared/nlp/tungks/ctopd_all'))
    p.add_argument('--runtime', type=pathlib.Path,
                   help='Installed prefix; default CTOPD_RUNTIME_PREFIX or /opt/ctopd-b200')
    p.add_argument('--algorithm', choices=['strict-top16', 'strict-full', 'simct', 'simct_paper', 'mp-atomic',
        'mp-fixed1', 'mp-fixed2', 'mp-fixed3', 'mp-fixed4', 'mp-fixed5',
        'mp-random1-2', 'mp-random2-5', 'mp-grass-chunk', 'mp-align', 'mp-align-new',
        'mp-trust-r','mp-trust-b','mp-trust-p',*H1B_ARMS], default='strict-top16')
    p.add_argument('--mp-chunk-source', choices=['xtoken', 'run'], default='xtoken')
    p.add_argument('--mp-chunk-run-length', type=int, default=2)
    p.add_argument('--mp-geometry', choices=['exact_head', 'diag'], default='exact_head')
    p.add_argument('--seed', type=int, default=42, help='Original launcher default, not a verified paper seed')
    p.add_argument('--micro-batch', type=int, choices=[1, 2, 4], default=4,
                   help='Original launcher uses 4; 1/2 retain global batch through accumulation')
    p.add_argument('--concurrency', type=int, default=8, help='Transport concurrency; samples/batch stay unchanged')
    p.add_argument('--eval-timeout', type=int, default=0, help='Seconds per evaluation request; 0 disables timeout')
    p.add_argument('--run-id', help='New output directory name; existing runs are never resumed')
    p.add_argument('--dry-run', action='store_true')
    return p


def build_plan(args):
    if (args.algorithm == 'mp-align-new' or args.algorithm.startswith('mp-trust-')) and args.mp_geometry != 'exact_head':
        raise ValueError('ALIGN-NEW/TRUST requires exact_head geometry')
    if args.concurrency < 1 or args.eval_timeout < 0 or args.mp_chunk_run_length < 1:
        raise ValueError('Concurrency must be positive and eval timeout nonnegative')
    run_id = args.run_id or (datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                             + '-' + args.algorithm + '-s' + str(args.seed))
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', run_id):
        raise ValueError('run-id must be a single directory name starting with a letter or digit')
    root = args.root.expanduser().resolve()
    runtime = (args.runtime or pathlib.Path(os.environ.get('CTOPD_RUNTIME_PREFIX', '/opt/ctopd-b200'))).expanduser().resolve()
    output = root / 'runs' / run_id
    flags = {
        'num_nodes': 1, 'num_gpus_per_node': 1, 'backend': 'fsdp2',
        'train_batch_size': 512, 'micro_train_batch_size': args.micro_batch,
        'learning_rate': '1e-6', 'lr_scheduler': 'constant', 'lr_warmup_ratio': 0.05,
        'adam_betas': '(0.9, 0.98)', 'weight_decay': 0, 'max_norm': 1.0,
        'num_epochs': 10, 'max_steps': 100, 'seed': args.seed,
        'save_path': str(output / 'checkpoints'), 'ckpt_path': str(output / 'unused-resume'),
        'bf16': True, 'gradient_checkpointing': True, 'enable_sleep': True,
        'use_dynamic_bsz': False,
        'student_name_or_path': str(root / 'models/Llama-3.2-3B-Instruct'),
        'teacher_name_or_path': str(root / 'models/Qwen2.5-7B-Instruct'),
        'enable_thinking': False,
        'rollout_batch_size': 512, 'rollout_num_engines': 1, 'rollout_tp_size': 1,
        'rollout_engine_concurrency': args.concurrency, 'rollout_mem_fraction_static': 0.6,
        'n_samples_per_prompt': 1, 'generate_max_len': 8192, 'temperature': 1.0, 'top_p': 0.95,
        'train_dataset_path': str(root / 'data/Train/math_code_20k/train.parquet'),
        'max_len': 10240, 'prompt_max_len': 2048, 'input_key': 'prompt',
        'label_key': 'reward_model.ground_truth', 'apply_chat_template': True,
        'preprocess_num_workers': 8, 'packing_samples': True,
        'eval_math_dataset_path': str(root / 'data/math500/test.jsonl'),
        'eval_math_max_tokens': 8192, 'eval_math_max_concurrent': args.concurrency,
        'eval_math_timeout': args.eval_timeout, 'eval_math_n_samples': 32,
        'eval_steps': 10, 'save_steps': 10, 'logging_steps': 1,
        'kd_ratio': 1.0, 'kd_loss_fn': 'rkl', 'kd_temperature': 1.0,
        'kd_algorithm': 'topk_byte_align_ctkd' if args.algorithm == 'strict-top16' else 'byte_align_ctkd',
        'ctkd_topk': 16, 'teacher_dp_size': 1, 'teacher_tp_size': 1, 'teacher_pp_size': 1,
        'teacher_mem_fraction_static': 0.6, 'teacher_offload_tags': 'all',
        'chunked_loss_size': 2048, 'use_wandb': False,
    }
    if args.algorithm in {'simct','simct_paper'}:
        flags.update(kd_algorithm=args.algorithm, simct_gh_mask_threshold=0.0)
    if args.algorithm.startswith('mp-'):
        name = args.algorithm[3:]
        family = 'fixed' if name.startswith('fixed') else 'random' if name.startswith('random') else name.replace('-', '_')
        flags['kd_algorithm'] = 'mp_opd_' + family
        if family == 'fixed':
            flags['mp_opd_fixed_span_length'] = int(name[-1])
        elif family == 'random':
            low, high = map(int, name[len('random'):].split('-'))
            flags.update(mp_opd_random_min_length=low, mp_opd_random_max_length=high,
                         mp_opd_random_seed=args.seed)
        if family in {'grass_chunk', 'align', 'align_new'}:
            flags.update(mp_opd_chunk_source=args.mp_chunk_source,
                         mp_opd_chunk_run_length=args.mp_chunk_run_length,
                         mp_opd_grass_geometry=args.mp_geometry)
        if family == 'align_new':
            flags.update(mp_opd_align_new_min_descent_fraction=0.1, mp_opd_align_new_eps_g=1e-12)
        if family.startswith('trust_'):
            flags.update(mp_opd_grass_geometry='exact_head',mp_opd_trust_eps_g=1e-12,mp_opd_trust_vocab_chunk=4096)
    if args.algorithm in H1B_ARMS:
        flags.update(kd_algorithm='h1b_span',h1b_arm=H1B_ARMS[args.algorithm],h1b_span_lambda=.5,
                     h1b_log_sample_rate=.05,h1b_dump_every='0,20,60,100',top_p=1.,eval_math_top_p=.95)
    command = ['bash', str(runtime / 'python-b200.sh'), '-u', '-m', 'kdflow.cli.train_kd_on_policy']
    for key, value in flags.items():
        command.extend(['--' + key, str(value)])
    return {'algorithm': args.algorithm, 'output': str(output), 'runtime': str(runtime),
            'source': str(SOURCE), 'flags': flags, 'command': command,
            'gradient_accumulation_microsteps': 512 // args.micro_batch,
            'expected_runtime_archive_sha256': ARCHIVE_SHA,
            'qualification': ('experimental H1b span arms; trunk gate pending; no B200/full-training qualification'
                              if args.algorithm in H1B_ARMS else
                              'paper-score SimCT reconstruction; no B200/full-training qualification'
                              if args.algorithm == 'simct_paper' else
                              'author-code SimCT port; no B200/full-training qualification'
                              if args.algorithm == 'simct' else
                              'experimental MP-OPD source port; no B200/full-training qualification'
                              if args.algorithm.startswith('mp-') else
                              'unvalidated single-B200 full workload; no metric-match claim'),
            'evaluation_scope': 'MATH500 accuracy over 32 responses/problem; not the full paper benchmark suite'}


def run_environment():
    env = os.environ.copy()
    env.update(RAY_ADDRESS='local', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONUNBUFFERED='1', PYTHONPATH=str(SOURCE))
    # Company HTTP proxies must not intercept local Ray/SGLang connections.
    bypass = [env.get('NO_PROXY', ''), env.get('no_proxy', ''), 'localhost', '127.0.0.1', '::1', socket.gethostname()]
    try:
        bypass.extend(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    env['NO_PROXY'] = env['no_proxy'] = ','.join(item for item in bypass if item)
    return env


def validate_files(plan):
    flags = plan['flags']
    required = [pathlib.Path(plan['runtime']) / 'python-b200.sh']
    required += [pathlib.Path(flags[key]) for key in ['train_dataset_path', 'eval_math_dataset_path']]
    for key in ['student_name_or_path', 'teacher_name_or_path']:
        model = pathlib.Path(flags[key])
        required += [model / 'config.json', model / 'tokenizer_config.json', model / 'tokenizer.json']
        index = model / 'model.safetensors.index.json'
        if index.is_file():
            shards = set(json.loads(index.read_text())['weight_map'].values())
            for shard in shards:
                if pathlib.PurePosixPath(shard).name != shard:
                    raise ValueError('Unexpected checkpoint shard path: ' + shard)
                required.append(model / shard)
        else:
            required.append(model / 'model.safetensors')
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if pathlib.Path(plan['output']).exists():
        raise FileExistsError('Choose a new run-id; refusing existing output: ' + plan['output'])


def main(argv=None):
    args = parser().parse_args(argv)
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    validate_files(plan)
    env = run_environment()
    commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip()
    if dirty:
        raise RuntimeError('Commit source changes before starting a recorded reproduction run')
    # This validates real model/data availability and GPU visibility, not a toy training workload.
    probe = '''import json, pathlib, torch, pyarrow.parquet as pq
assert torch.cuda.device_count() == 1, 'Expose exactly the one GPU allocated to this run'
name = torch.cuda.get_device_name(0)
assert 'B200' in name, 'This launcher targets a B200'
train, evaluation = __import__('sys').argv[1:]
metadata = pq.read_metadata(train)
assert metadata.num_rows == 20000, 'Expected authors source pool of 20,000 prompts before tokenizer filtering'
assert {'prompt', 'reward_model'} <= set(pq.read_schema(train).names)
rows = [json.loads(line) for line in pathlib.Path(evaluation).read_text().splitlines() if line.strip()]
assert len(rows) == 500 and all({'problem', 'answer'} <= row.keys() for row in rows)
print(json.dumps({'gpu': name, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'source_rows': metadata.num_rows, 'math500_rows': len(rows)}))
'''
    result = subprocess.check_output(['bash', str(pathlib.Path(plan['runtime']) / 'python-b200.sh'),
        '-c', probe, plan['flags']['train_dataset_path'], plan['flags']['eval_math_dataset_path']],
        env=env, text=True)
    plan['source_commit'] = commit
    plan['preflight'] = json.loads(result.strip().splitlines()[-1])
    runtime_receipt = pathlib.Path(plan['runtime']) / '.ctopd-runtime.json'
    identity = json.loads(runtime_receipt.read_text()) if runtime_receipt.is_file() else {}
    plan['installed_runtime_identity'] = {
        'archive_sha256': identity.get('archive_sha256'),
        'compatibility_overlay_sha256': identity.get('compatibility_overlay', {}).get('sha256'),
        'receipt_present': runtime_receipt.is_file(),
    }
    plan['input_sha256'] = {}
    for key in ['train_dataset_path', 'eval_math_dataset_path']:
        path = pathlib.Path(plan['flags'][key])
        plan['input_sha256'][key] = hashlib.sha256(path.read_bytes()).hexdigest()
    output = pathlib.Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'launch.json').write_text(json.dumps(plan, indent=2) + '\n')
    (output / 'command.sh').write_text(shlex.join(plan['command']) + '\n')
    print('RUN_DIRECTORY=' + str(output), flush=True)
    os.chdir(SOURCE)
    os.execvpe(plan['command'][0], plan['command'], env)


if __name__ == '__main__':
    main()
