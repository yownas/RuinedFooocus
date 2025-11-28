import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--settings", type=str, default=None, help="Select setting")
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
parser.add_argument("--mcp", action="store_true", help="Start MCP server.")
parser.add_argument("--nobrowser", action="store_true", help="Do not launch in browser.")
parser.add_argument("--gpu-device-id", type=int, default=None, metavar="DEVICE_ID")
parser.add_argument("--offline", action="store_true", help="Skip update-check during startup.")
parser.add_argument("--iINSTallLEDmYOwNPaCKaGeS", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--language", type=str, default='en', help="UI language.")
parser.add_argument("--clean-cache", action="store_true", help="Purge old cache before starting.")

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

fp_group = parser.add_mutually_exclusive_group()
fp_group.add_argument("--force-fp32", action="store_true", help="Force fp32 (If this makes your GPU work better please report it).")
fp_group.add_argument("--force-fp16", action="store_true", help="Force fp16.")

fpunet_group = parser.add_mutually_exclusive_group()
fpunet_group.add_argument("--fp32-unet", action="store_true", help="Run the diffusion model in fp32.")
fpunet_group.add_argument("--fp64-unet", action="store_true", help="Run the diffusion model in fp64.")
fpunet_group.add_argument("--bf16-unet", action="store_true", help="Run the diffusion model in bf16.")
fpunet_group.add_argument("--fp16-unet", action="store_true", help="Run the diffusion model in fp16")
fpunet_group.add_argument("--fp8_e4m3fn-unet", action="store_true", help="Store unet weights in fp8_e4m3fn.")
fpunet_group.add_argument("--fp8_e5m2-unet", action="store_true", help="Store unet weights in fp8_e5m2.")
fpunet_group.add_argument("--fp8_e8m0fnu-unet", action="store_true", help="Store unet weights in fp8_e8m0fnu.")

fpvae_group = parser.add_mutually_exclusive_group()
fpvae_group.add_argument("--fp16-vae", action="store_true", help="Run the VAE in fp16, might cause black images.")
fpvae_group.add_argument("--fp32-vae", action="store_true", help="Run the VAE in full precision fp32.")
fpvae_group.add_argument("--bf16-vae", action="store_true", help="Run the VAE in bf16.")

parser.add_argument("--cpu-vae", action="store_true", help="Run the VAE on the CPU.")

fpte_group = parser.add_mutually_exclusive_group()
fpte_group.add_argument("--fp8_e4m3fn-text-enc", action="store_true", help="Store text encoder weights in fp8 (e4m3fn variant).")
fpte_group.add_argument("--fp8_e5m2-text-enc", action="store_true", help="Store text encoder weights in fp8 (e5m2 variant).")
fpte_group.add_argument("--fp16-text-enc", action="store_true", help="Store text encoder weights in fp16.")
fpte_group.add_argument("--fp32-text-enc", action="store_true", help="Store text encoder weights in fp32.")
fpte_group.add_argument("--bf16-text-enc", action="store_true", help="Store text encoder weights in bf16.")

args = parser.parse_args()

