#!/bin/sh

cat <<EOF
Some systems fail to install the insightface module, which is why this is not
part of the default install.

Make sure you have a proper pyenv/virtualenv activated.

Press enter to continue...
EOF
read junk

pip install insightface==0.7.3 gfpgan==1.3.8 git+https://github.com/rodjjo/filterpy.git --require-virtualenv

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
