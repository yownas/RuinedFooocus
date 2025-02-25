import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=None, help="Set the listen port.")
parser.add_argument(
    "--share", action="store_true", help="Set whether to share on Gradio."
)
parser.add_argument("--auth", type=str, help="Set credentials username/password.")
parser.add_argument(
    "--listen",
    type=str,
    default=None,
    metavar="IP",
    nargs="?",
    const="0.0.0.0",
    help="Set the listen interface.",
)
parser.add_argument(
    "--nobrowser", action="store_true", help="Do not launch in browser."
)
parser.add_argument("--gpu-device-id", type=int, default=None, metavar="DEVICE_ID")
parser.add_argument("--offline", action="store_true", help=argparse.SUPPRESS)

# ComfyUI arguments
parser.add_argument("--directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1, help="Use torch-directml.")
vram_group = parser.add_mutually_exclusive_group()
vram_group.add_argument("--gpu-only", action="store_true", help="Store and run everything (text encoders/CLIP models, etc... on the GPU).")
vram_group.add_argument("--highvram", action="store_true", help="By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory.")
vram_group.add_argument("--normalvram", action="store_true", help="Used to force normal vram use if lowvram gets automatically enabled.")
vram_group.add_argument("--lowvram", action="store_true", help="Split the unet in parts to use less vram.")
vram_group.add_argument("--novram", action="store_true", help="When lowvram isn't enough.")
vram_group.add_argument("--cpu", action="store_true", help="To use the CPU for everything (slow).")
parser.add_argument("--reserve-vram", type=float, default=None, help="Set the amount of vram in GB you want to reserve for use by your OS/other software. By default some amount is reserved depending on your OS.")



args = parser.parse_args()

