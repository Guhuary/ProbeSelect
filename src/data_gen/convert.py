
# Monkey-patch to fix missing ImageNetInfo
try:
    from timm.data import ImageNetInfo, infer_imagenet_subset # type: ignore
except ImportError:
    # Create a dummy class if not exists
    class ImageNetInfo:
        def __init__(self, *args, **kwargs):
            pass
    class infer_imagenet_subset:
        def __init__(self, *args, **kwargs):
            pass
    import timm.data
    timm.data.ImageNetInfo = ImageNetInfo # type: ignore
    timm.data.infer_imagenet_subset = infer_imagenet_subset # type: ignore

import numpy as np
import torch

import sys
import os

from PIL import Image
from functools import partial

from tqdm import tqdm, trange
import argparse
from evaluator import *
from utils import *

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert raw DiQE prompt-image tensors into scored feature tensors."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (inclusive). Default: 0.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=10000,
        help="End index (exclusive). Default: 10000.",
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default="/mnt/sharedata/hdd/users/guohl/datasets/DiQE/cocotrain_fullcap-numimgs_alltime_perprompt5_pca48_imgcaponly",
        help="Input directory containing source .pt files.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="/mnt/sharedata/hdd/users/guohl/datasets/DiQE/cocotrain_fullcap-numimgs_alltime_perprompt5_pca48",
        help="Output directory to save converted .pt files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help='Torch device used for evaluators, e.g. "cuda:0" or "cpu".',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
        help="Random seed used to initialize the torch generator.",
    )
    return parser


def build_evaluators(eval_device: str):
    evaluators = {
        "clip16": CLIP16Evaluator(device=eval_device),
        "clip32": CLIP32Evaluator(device=eval_device),
        "as": AestheticScoreEvaluator(device=eval_device),
        "pick": PickScoreEvaluator(device=eval_device),
    }

    text_embeder = {
        "clip16_t": evaluators["clip16"].get_embedding,
        "clip32_t": evaluators["clip32"].get_embedding,
        "pick_t": evaluators["pick"].get_embedding,
    }

    blip_eval = BLIP_Evaluator(device=eval_device)
    imgreward_eval = ImgReward_Evaluator(device=eval_device)
    hpsv20_eval = HPS_Evaluator(device=eval_device, hps_version="v2.0")
    hpsv21_eval = HPS_Evaluator(device=eval_device, hps_version="v2.1")
    score_embed_both_evaluator = {
        "BLIP_ITC": blip_eval.get_itc_score_and_embedding,
        "BLIP_ITM": blip_eval.get_itm_score_and_embedding,
        "ImgReward": imgreward_eval.get_score_and_embedding,
        "HPSv20": hpsv20_eval.get_score_and_embedding,
        "HPSv21": hpsv21_eval.get_score_and_embedding,
    }
    return evaluators, text_embeder, score_embed_both_evaluator


def run_conversion(args: argparse.Namespace) -> None:
    if args.start >= args.end:
        raise ValueError(f"`start` must be smaller than `end`, got {args.start} >= {args.end}")

    os.makedirs(args.out_path, exist_ok=True)
    _ = torch.Generator(torch.device(args.device)).manual_seed(args.seed)

    evaluators, text_embeder, score_embed_both_evaluator = build_evaluators(args.device)
    print(f"Converting indices in [{args.start}, {args.end}) on device={args.device}")
    print(f"Input: {args.raw_path}")
    print(f"Output: {args.out_path}")

    for i in trange(args.start, args.end):
        current_path = os.path.join(args.raw_path, f"{i}.pt")
        output_path = os.path.join(args.out_path, f"{i}.pt")
        if os.path.exists(output_path) or not os.path.exists(current_path):
            continue

        output_dict = torch.load(current_path, weights_only=False)
        prompt = output_dict["prompt"]
        images = output_dict["images"]

        for name, evaluator in evaluators.items():
            output_dict[name] = evaluator.get_score(prompt, images).cpu().detach()
        # output_dict["ICT_HP"] = output_dict["ICT"] * output_dict["HP"]

        for name, text_embed in text_embeder.items():
            output_dict[name] = text_embed(prompt).cpu().detach()
        # output_dict["ICT_HP_t"] = output_dict["ICT_t"]

        for name, score_embed_both in score_embed_both_evaluator.items():
            score, embedding = score_embed_both(prompt, images)
            output_dict[name] = score.cpu().detach()
            output_dict[name + "_t"] = embedding.cpu().detach()

        del output_dict["images"]
        del output_dict["prompt"]
        torch.save(output_dict, output_path)


def main() -> None:
    args = build_parser().parse_args()
    run_conversion(args)


if __name__ == "__main__":
    main()
    