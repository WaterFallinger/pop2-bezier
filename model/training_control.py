"""Use the existing cooperative save/pause/resume interface."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['status', 'save', 'pause', 'resume'])
    parser.add_argument('--control-dir', type=Path, help='Explicit trainer control directory; default follows the local run')
    args = parser.parse_args()
    root = args.control_dir or Path(os.environ.get('V33_TRAINING_CONTROL_DIR', str(Path(__file__).resolve().parent / 'outputs' / 'control')))
    status = json.loads((root / 'status.json').read_text(encoding='utf-8'))
    if args.action == 'status':
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return
    request_path = root / 'request.json'
    if request_path.exists():
        pending = json.loads(request_path.read_text(encoding='utf-8'))
        if (pending.get('run_id') == status['run_id']
                and int(pending.get('sequence', 0)) > int(status.get('last_completed_sequence', 0))):
            raise RuntimeError('An earlier request is still pending; do not overwrite it.')
    request = {'run_id': status['run_id'], 'sequence': time.time_ns(), 'action': args.action}
    temporary = request_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(request), encoding='utf-8')
    os.replace(temporary, request_path)
    print(json.dumps(request))


if __name__ == '__main__':
    main()
