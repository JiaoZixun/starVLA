"""Run the official QwenOFT adapter and track real results with tqdm."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time

from tqdm import tqdm

from robodojo_eval_assets import format_bytes, inspect_eval_assets


def read_result(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def stop_process(process):
    if process is None:
        return
    # Only signal this evaluation's process group, including its model server.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', nargs='+', help='Default: official 42 tasks')
    parser.add_argument('--output', type=Path, help='Reuse this directory to resume')
    parser.add_argument('--check-only', action='store_true', help='Validate without simulation')
    args = parser.parse_args()
    bench = Path(os.environ['ROBODOJO_PATH']).resolve()
    policy = bench / 'XPolicyLab/policy/starVLA'
    source = policy / 'source_starvla'
    checkpoint = Path(os.environ.get('STARVLA_HF_ROOT', '/home/data/robodojo_runs/qwen_oft')).resolve() / 'oft'
    episodes = int(os.environ['EPISODES'])
    num_envs = int(os.environ['NUM_ENVS'])
    seed = int(os.environ['LAYOUT_SEED'])
    gpu = int(os.environ['GPU_ID'])
    if episodes < 1 or num_envs < 1 or seed < 0 or gpu < 0:
        parser.error('EPISODES/NUM_ENVS must be positive; seed/GPU must be nonnegative')
    env = os.environ.copy()
    env.pop('STARVLA_BASE_VLM', None)
    env.pop('EVAL_ENV_TYPE', None)
    env['PYTHONPATH'] = os.pathsep.join([str(source), str(bench), str(bench / 'XPolicyLab')])
    env['STARVLA_HF_VERIFY_ONLY'] = '1'
    env['STARVLA_HF_SKIP_WEIGHT_HASH'] = '1'
    env['STARVLA_ROBODOJO_NUM_ENVS'] = str(num_envs)
    # Same setting already used by the local RoboDojo installation.
    env.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')

    spec = importlib.util.spec_from_file_location('task_inventory', bench / 'scripts/internal/task_inventory.py')
    inventory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inventory)
    official = [t for group in inventory.DIMENSION_TASKS.values() for t in group]
    tasks = args.tasks or official
    if len(tasks) != len(set(tasks)) or any(t not in official for t in tasks):
        parser.error('--tasks must contain unique official task names')
    try:
        num_envs_overrides = {
            str(task): int(value)
            for task, value in json.loads(
                os.environ.get('STARVLA_ROBODOJO_TASK_NUM_ENVS_OVERRIDES', '{}')
            ).items()
        }
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f'invalid STARVLA_ROBODOJO_TASK_NUM_ENVS_OVERRIDES: {exc}')
    if any(task not in official or value < 1 or value > num_envs
           for task, value in num_envs_overrides.items()):
        parser.error('task environment overrides must name official tasks and be in [1, NUM_ENVS]')
    layouts = bench / 'Assets/Eval_Layout/RoboDojo/arx_x5' / str(seed)
    for task in tasks:
        if not (bench / 'task/RoboDojo/config' / (task + '.yml')).is_file():
            raise FileNotFoundError(f'Task config missing: {task}')
        if len(list(layouts.glob(task + '_[0-9]*.json'))) < episodes:
            raise ValueError(f'Insufficient official layouts for {task}: {layouts}')
    asset_report = inspect_eval_assets(bench, tasks, episodes, seed)
    if asset_report['missing'] or asset_report['issues']:
        asset_prepare = Path(__file__).with_name('prepare_robodojo_eval_assets.py')
        download_command = shlex.join([
            sys.executable, str(asset_prepare), '--seed', str(seed), '--episodes', str(episodes),
            '--tasks', *tasks, '--download',
        ])
        details = [f"  missing/corrupt: {path}" for path in asset_report['missing'][:20]]
        details.extend(f"  issue: {issue}" for issue in asset_report['issues'][:20])
        more = max(0, len(asset_report['missing']) - 20) + max(0, len(asset_report['issues']) - 20)
        if more:
            details.append(f'  ... and {more} more entries')
        detail_text = '\n'.join(details)
        raise RuntimeError(
            'Official RoboDojo assets required by these layouts are incomplete. '
            f"Missing/corrupt files: {len(asset_report['missing'])}; "
            f"unresolved issues: {len(asset_report['issues'])}; "
            f"estimated download: {format_bytes(asset_report['estimated_bytes'])}.\n"
            f'{detail_text}\n'
            'Fetch the exact assets with:\n'
            f'  {download_command}'
        )

    print('Checking official checkpoint, normalization and size...', flush=True)
    subprocess.run([sys.executable, str(policy / 'scripts/prepare_hf_checkpoint.py'),
                    '--variant', 'oft', '--output-dir', str(checkpoint), '--verify-only'],
                   env=env, cwd=source, check=True)
    subprocess.run([os.environ['SIM_ENV'] + '/bin/python',
                    str(policy / 'scripts/check_isaac_runtime.py')], check=True)
    subprocess.run([sys.executable, '-c',
                    'from deployment.model_server.policy_wrapper import PolicyServerWrapper; '
                    'from starVLA.model.framework.base_framework import _auto_import_framework_modules; '
                    '_auto_import_framework_modules(); print("StarVLA imports OK")'],
                   cwd=source, env=env, check=True)
    subprocess.run([sys.executable, str(source / 'deployment/model_server/server_policy.py'),
                    '--help'], cwd=source, env=env, check=True, stdout=subprocess.DEVNULL)
    output = (args.output or Path(os.environ['STARVLA_HF_ROOT']) /
              ('eval_' + time.strftime('%Y%m%d_%H%M%S'))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    manifest = dict(tasks=tasks, episodes=episodes, num_envs=num_envs, seed=seed,
                    gpu=gpu, checkpoint=str(checkpoint), benchmark=str(bench),
                    policy_python=sys.executable, simulator=os.environ['SIM_ENV'],
                    num_envs_overrides=num_envs_overrides,
                    layouts='Official randomized layout queue, numeric order',
                    base_vlm='Qwen/Qwen3-VL-4B-Instruct', run_id=output.name)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous != manifest:
            raise ValueError('Resume settings differ from manifest.json; use a new --output')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Output: {output}\nTasks: {len(tasks)}, layouts/task: {episodes}, parallel envs: {num_envs}', flush=True)
    print('ETA becomes available after the first completed layout batch; startup/downloads are logged per task.', flush=True)
    if args.check_only:
        print('Preflight passed. No simulation started.')
        return

    # Upstream initializes from the base VLM before strictly restoring all
    # fine-tuned parameters. Download before the server's connection timeout.
    from huggingface_hub import HfApi, snapshot_download

    base_model = 'Qwen/Qwen3-VL-4B-Instruct'
    base_revision_path = output / 'base_revision.txt'
    if base_revision_path.exists():
        base_revision = base_revision_path.read_text().strip()
    else:
        base_revision = HfApi().model_info(base_model).sha
        base_revision_path.write_text(base_revision + '\n')
    print('Preparing official Qwen3-VL-4B base cache (first run only; download progress follows).', flush=True)
    base_path = Path(os.environ['STARVLA_HF_ROOT']) / 'Qwen3-VL-4B-Instruct' / base_revision
    snapshot_download(repo_id=base_model, revision=base_revision, local_dir=base_path,
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'tokenizer*'])
    env['STARVLA_BASE_VLM'] = str(base_path.resolve())

    rows = []
    process = None
    with tqdm(total=len(tasks) * episodes, unit='episode', dynamic_ncols=True,
              smoothing=0, desc='QwenOFT', mininterval=1) as progress:
        try:
            for task in tasks:
                run_id = manifest['run_id'] + '-' + task
                result_path = (bench / 'eval_result/RoboDojo' / task / 'starVLA/arx_x5' /
                               f'{seed}_ckpt_name=hf_qwenoft_100k,action_type=joint' / run_id / '_result.json')
                log_path = output / (task + '.log')
                data = read_result(result_path)
                counted = min(episodes, int(data.get('eval_time', 0)))
                progress.update(counted)
                rc = 0
                if counted < episodes:
                    task_env = dict(env, ROBODOJO_RUN_ID=run_id)
                    task_env['STARVLA_ROBODOJO_NUM_ENVS'] = str(
                        num_envs_overrides.get(task, num_envs)
                    )
                    command = ['bash', str(policy / 'scripts/eval_hf_robodojo.sh'), 'oft', task,
                               str(seed), str(gpu), str(gpu), os.environ['POLICY_ENV'],
                               os.environ['SIM_ENV'], str(episodes)]
                    progress.set_description(task)
                    tqdm.write(f'Running {task}; log: {log_path}')
                    with log_path.open('a') as log:
                        process = subprocess.Popen(command, cwd=bench, env=task_env,
                                                   stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                        while process.poll() is None:
                            data = read_result(result_path)
                            done = min(episodes, int(data.get('eval_time', 0)))
                            if done > counted:
                                progress.update(done - counted)
                                counted = done
                            progress.set_postfix(completed=f'{counted}/{episodes}',
                                                 SR=f"{100 * data.get('success_rate', 0):.1f}%",
                                                 refresh=True)
                            time.sleep(2)
                        rc = process.returncode
                        stop_process(process)
                        process = None
                    data = read_result(result_path)
                    done = min(episodes, int(data.get('eval_time', 0)))
                    progress.update(max(0, done - counted))
                    counted = done
                row = dict(task=task, episodes=counted,
                           success_rate_pct=100 * data['success_rate'] if counted else '',
                           score=data.get('score', ''), result=str(result_path), log=str(log_path),
                           status='complete' if counted == episodes and rc == 0 else 'incomplete')
                rows.append(row)
                if data:
                    (output / (task + '.result.json')).write_text(json.dumps(data, indent=2) + '\n')
                with (output / 'summary.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
                if row['status'] != 'complete':
                    raise RuntimeError(f'{task}: exit={rc}, completed={counted}/{episodes}. '
                                       f'Inspect {log_path}; resume with --output {output}')
        finally:
            stop_process(process)
    print(f'Completed. Results: {output / "summary.csv"}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped. Reuse --output to resume completed work.', file=sys.stderr)
        sys.exit(130)
