import os
import sys
os.chdir(sys.path[0])
from argparse import ArgumentParser
from model.apis import inference_segmentor, init_segmentor, show_result_pyplot
from model.core.evaluation import get_palette
FILENAME="2076.png"
CHECKPOINT='./exps/EODiFusionV1/iter_100000.pth'

def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default=f"/home/BlueDisk/Dataset/FusionDataset/RGBT/OverExposure/test/vi/{FILENAME}",
                        help='rgb file')
    parser.add_argument('--ir',default=f"/home/BlueDisk/Dataset/FusionDataset/RGBT/OverExposure/test/ir/{FILENAME}",
                        help='ir file')
    parser.add_argument('--fusion_img',default=f"./out/fusion/{FILENAME}",
                         help='Image file')
    parser.add_argument('--config', default="configs/config.py",
                        help='Config file')
    parser.add_argument('--checkpoint', default=CHECKPOINT,
                        help='Checkpoint file')
    parser.add_argument('--segout', default=f"./out/seg/{FILENAME}", 
                        help='Path to output file')
    parser.add_argument('--device', default='cuda:1', 
                        help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='msrs',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    """build the model from a config file and a checkpoint file"""
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    """test a single image"""
    result = inference_segmentor(model, args.img,args.ir)
    show_result_pyplot( 
        model,
        args.img,
        result,
        get_palette(args.palette),
        opacity=args.opacity,
        out_file=args.segout)

if __name__ == '__main__':
    main()
