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


args = parser.parse_args()

