"""Run a trained model on images and save or show the result.

    python predict.py --weights runs/my_run/weights/best.pt --source imgs/my_photos
    python predict.py --weights runs/my_run/weights/best.pt --source imgs/my_photos --show

--save (the default) writes annotated images to runs/predict/ and needs no
display, so it also works over SSH. --show opens a window instead and requires
X11 forwarding (setup.sh handles this on the host).
"""

import argparse

from common import RUNS_DIR, check_cuda, die, resolve

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def parse_args():
    p = argparse.ArgumentParser(
        description='Run inference with a trained YOLO model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--weights', required=True, help='path to a .pt checkpoint, e.g. runs/my_run/weights/best.pt')
    p.add_argument('--source', required=True, help='image file, folder of images, or a camera index like 0')
    p.add_argument('--conf', type=float, default=0.25, help='confidence threshold (default: 0.25)')
    p.add_argument('--device', default='0', help='GPU index or cpu (default: 0)')
    p.add_argument('--imgsz', type=int, help='inference image size (default: the size used for training)')
    p.add_argument('--line-width', type=int, default=2, help='box line width (default: 2)')
    p.add_argument('--hide-conf', action='store_true', help='do not draw confidence values')
    p.add_argument('--show', action='store_true', help='open a window per image instead of saving (needs X11)')
    p.add_argument('--name', default='predict', help='output folder name under runs/ (default: predict)')
    return p.parse_args()


def collect_images(source):
    """Return the list of images to run on, or None for a camera/stream source."""
    if source.isdigit():
        return None
    path = resolve(source)
    if path.is_file():
        return [path]
    if path.is_dir():
        images = sorted(p for p in path.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            die(f'No images found in {path}',
                f'Supported suffixes: {", ".join(sorted(IMAGE_SUFFIXES))}')
        return images
    die(f'Source not found: {path}',
        'Pass an image, a folder of images, or a camera index such as 0.')


def show_loop(model, images, predict_kwargs, plot_kwargs):
    """Display results one by one; any key advances, q or a closed window quits."""
    import cv2

    window = 'Detection Result'
    for img in images:
        results = model.predict(source=img, **predict_kwargs)
        print(f'{img.name}: {len(results[0])} objects, {results[0].speed}')
        try:
            cv2.imshow(window, results[0].plot(**plot_kwargs))
        except cv2.error as e:
            die('Cannot open a display window.',
                str(e),
                'Run without --show to save the images instead, or run',
                '  xhost +local:docker',
                'on the host before starting the container.')
        print('Press any key in the image window to continue (q to quit)...')
        if cv2.waitKey(0) in (ord('q'), 27):
            break
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            print('Window closed by user. Exiting.')
            break
    cv2.destroyAllWindows()


def main():
    cli = parse_args()

    weights = resolve(cli.weights)
    if not weights.is_file():
        found = sorted(str(p.relative_to(RUNS_DIR.parent)) for p in RUNS_DIR.glob('*/weights/best.pt'))
        die(f'Weights not found: {weights}',
            *(['Checkpoints in runs/:'] + [f'  - {f}' for f in found] if found else
              ['No runs/*/weights/best.pt yet — train a model first.']))

    check_cuda(cli.device)
    images = collect_images(cli.source)

    predict_kwargs = {'conf': cli.conf, 'device': cli.device, 'verbose': False}
    if cli.imgsz:
        predict_kwargs['imgsz'] = cli.imgsz
    plot_kwargs = {'line_width': cli.line_width, 'conf': not cli.hide_conf}

    from ultralytics import YOLO

    model = YOLO(str(weights))

    if cli.show:
        if images is None:
            die('--show does not support a camera source here.',
                'Use a folder of images, or the Ultralytics CLI for live video.')
        show_loop(model, images, predict_kwargs, plot_kwargs)
        return

    out_dir = RUNS_DIR / cli.name
    model.predict(
        source=int(cli.source) if images is None else str(resolve(cli.source)),
        save=True,
        project=str(RUNS_DIR),
        name=cli.name,
        line_width=cli.line_width,
        show_conf=not cli.hide_conf,  # `conf` is the threshold here, not a plot option
        **predict_kwargs,
    )
    print(f'\nAnnotated images saved under {out_dir}* (a suffix is added if it already existed).')


if __name__ == '__main__':
    main()
