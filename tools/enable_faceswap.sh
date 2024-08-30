#!/bin/sh

cat <<EOF
Some systems fail to install the insightface module, which is why this is not
part of the default install.

Press enter to continue...
EOF
read junk

. $(dirname "$(readlink -f "$0")")/../venv/bin/activate

pip install insightface==0.7.3 gfpgan==1.3.8 --require-virtualenv

cat <<EOF
pip done...


Now add this to settings/powerup.json

  "Faceswap": {
    "type": "faceswap"
  }

Download the required models to models\faceswap\:

https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
and inswapper_128.onnx from where you can find it

You can find more info on the RuinedFooocus Github or Discord server.
EOF
