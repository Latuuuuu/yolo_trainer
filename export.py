"""Export a trained checkpoint to a deployment format (ONNX by default).

    python export.py --weights runs/my_run/weights/best.pt
    python export.py --weights runs/my_run/weights/best.pt --format engine --imgsz 960

The exported file is written next to the .pt file.
Format list: https://docs.ultralytics.com/modes/export/#export-formats
"""

import argparse

from common import RUNS_DIR, check_cuda, die, resolve


def parse_args():
    p = argparse.ArgumentParser(
        description='Export a YOLO checkpoint for deployment.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--weights', required=True, help='path to a .pt checkpoint')
    p.add_argument('--format', default='onnx', help='export format (default: onnx; e.g. engine, torchscript, openvino)')
    p.add_argument('--imgsz', type=int, default=640, help='input size baked into the exported model (default: 640)')
    p.add_argument('--opset', type=int, default=11, help='ONNX opset version (default: 11)')
    p.add_argument('--device', default='0', help='GPU index or cpu (default: 0)')
    p.add_argument('--half', action='store_true', help='export in FP16 (smaller and faster, GPU only)')
    p.add_argument('--no-simplify', action='store_true', help='skip ONNX graph simplification')
    return p.parse_args()


def main():
    cli = parse_args()

    weights = resolve(cli.weights)
    if not weights.is_file():
        found = sorted(str(p.relative_to(RUNS_DIR.parent)) for p in RUNS_DIR.glob('*/weights/best.pt'))
        die(f'Weights not found: {weights}',
            *(['Checkpoints in runs/:'] + [f'  - {f}' for f in found] if found else
              ['No runs/*/weights/best.pt yet — train a model first.']))

    check_cuda(cli.device)

    from ultralytics import YOLO

    kwargs = {
        'format': cli.format,
        'imgsz': cli.imgsz,
        'device': cli.device,
        'half': cli.half,
        'simplify': not cli.no_simplify,
    }
    if cli.format == 'onnx':
        kwargs['opset'] = cli.opset

    print(f'[info] exporting {weights} to {cli.format} at imgsz={cli.imgsz}')
    YOLO(str(weights)).export(**kwargs)


if __name__ == '__main__':
    main()
