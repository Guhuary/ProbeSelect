import argparse
import os
from typing import Any, cast

import torch
from diffusers import StableDiffusion3Pipeline  # type: ignore
from tqdm import trange

from utils import ActivationGrabber_SD3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SD3-M images and save intermediate PCA features."
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start ratio of batches to process, in [0, 1].",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=1.0,
        help="End ratio of batches to process, in [0, 1].",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of prompts per generation batch.",
    )
    parser.add_argument(
        "--num-imgs-per-prompt",
        type=int,
        default=5,
        help="How many times to duplicate each prompt.",
    )
    parser.add_argument(
        "--num-infer-steps",
        type=int,
        default=28,
        help="Diffusion sampling steps for each generation call.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
        help="Random seed used by torch.Generator.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=100000,
        help="Maximum number of prompts loaded from prompt file.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/mnt/sharedata/ssd_large/users/guohl/pretrained_models/stabilityai/stable-diffusion-3.5-medium",
        help="Path to Stable Diffusion 3.5 Medium checkpoint.",
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        default="/mnt/sharedata/ssd_large/users/guohl/Gen/datasets/MS COCO annotations/train2017_captions.txt",
        help="Path to text file containing prompts (one per line).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Output directory for .pt files. If omitted, build from num-imgs-per-prompt.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip output file if it already exists.",
    )
    parser.add_argument(
        "--record-start-ratio",
        type=float,
        default=0.05,
        help="Only record features when step/num_infer_steps >= this value.",
    )
    parser.add_argument(
        "--record-step-interval",
        type=int,
        default=2,
        help="Record feature every N denoising steps.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 <= args.start <= 1.0 and 0.0 <= args.end <= 1.0):
        raise ValueError("--start and --end must both be in [0, 1].")
    if args.start >= args.end:
        raise ValueError("--start must be smaller than --end.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_imgs_per_prompt <= 0:
        raise ValueError("--num-imgs-per-prompt must be positive.")
    if args.num_infer_steps <= 0:
        raise ValueError("--num-infer-steps must be positive.")
    if args.max_prompts <= 0:
        raise ValueError("--max-prompts must be positive.")
    if not (0.0 <= args.record_start_ratio <= 1.0):
        raise ValueError("--record-start-ratio must be in [0, 1].")
    if args.record_step_interval <= 0:
        raise ValueError("--record-step-interval must be positive.")


def load_prompts(prompt_path: str, max_prompts: int, num_imgs_per_prompt: int) -> list[str]:
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines()[:max_prompts]]
    prompts = [p for p in prompts if p]
    return prompts * num_imgs_per_prompt


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.save_path is None:
        save_path = (
            "/mnt/sharedata/hdd/users/guohl/datasets/DiQE/"
            f"SD3_M_cocotrain_fullcap-numimgs_alltime_perprompt{args.num_imgs_per_prompt}_pca48_imgcaponly"
        )
    else:
        save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)

    all_prompts = load_prompts(args.prompt_path, args.max_prompts, args.num_imgs_per_prompt)
    total_batches = len(all_prompts) // args.batch_size
    start_batch = int(args.start * total_batches)
    end_batch = int(args.end * total_batches)
    print(
        f"start={args.start} end={args.end} total_prompts={len(all_prompts)} "
        f"batch_size={args.batch_size} start_batch={start_batch} end_batch={end_batch}"
    )

    grabber = ActivationGrabber_SD3(pipe.transformer)
    try:
        for batch_idx in trange(start_batch, end_batch):
            output_path = os.path.join(save_path, f"{batch_idx}.pt")
            if args.skip_existing and os.path.exists(output_path):
                continue

            batch_prompts = all_prompts[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
            records: list[dict[str, Any]] = []

            def _on_step(
                _pipe: StableDiffusion3Pipeline,
                step_index: int,
                _timestep: int,
                _callback_kwargs: dict[str, Any],
            ) -> dict[str, Any]:
                if (
                    step_index / args.num_infer_steps >= args.record_start_ratio
                    and step_index % args.record_step_interval == 0
                ):
                    pooled = grabber.get_pca_feature()
                    records.append(
                        {
                            "t": step_index / args.num_infer_steps,
                            "feat": pooled,
                        }
                    )
                return {}

            with torch.no_grad():
                images = pipe(
                    batch_prompts,
                    num_inference_steps=args.num_infer_steps,
                    callback_on_step_end=cast(Any, _on_step),
                    generator=generator,
                ).images  # pyright: ignore[reportAttributeAccessIssue]

            output_dict = {
                "records": records,
                "images": images,
                "prompt": batch_prompts,
            }
            torch.save(output_dict, output_path)
    finally:
        grabber.close()


if __name__ == "__main__":
    main()





