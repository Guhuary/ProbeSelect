import argparse
import torch
from src.new_pipe.SD3 import ProbeSelect_SD3

def parse_args():
    parser = argparse.ArgumentParser(description="ProbeSelect SD3L Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/sharedata/ssd_large/users/guohl/pretrained_models/stabilityai/stable-diffusion-3.5-large",
        help="Path to the pretrained SD3L model",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="workdir/SD3L_IR_ema.ckpt",
        help="Path to the ProbeSelect checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (e.g., cuda:0, cuda:1, cpu)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default="a beautiful landscape",
        help="Text prompt(s) for image generation",
    )
    parser.add_argument(
        "--enable_probe_select",
        action="store_true",
        default=True,
        help="Enable probe selection",
    )
    parser.add_argument(
        "--num_select",
        type=int,
        default=1,
        help="Number of images to select",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=5,
        help="Number of images to generate per prompt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save generated images",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    pipe = ProbeSelect_SD3.from_pretrained(
        args.model_path,
        args.ckpt_path,
        torch_dtype=torch.bfloat16
    ).to(device)

    result = pipe(
        args.prompt,
        enable_probe_select=args.enable_probe_select,
        num_select=args.num_select,
        num_images_per_prompt=args.num_images_per_prompt,
    )

    # Save images
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    for i, image in enumerate(result.images):  # pyright: ignore[reportAttributeAccessIssue]
        image.save(os.path.join(args.output_dir, f"image_{i}.png"))
        print(f"Saved: {args.output_dir}/image_{i}.png")


if __name__ == "__main__":
    main()
