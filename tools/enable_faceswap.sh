#!/bin/sh

. $(dirname "$(readlink -f "$0")")/../venv/bin/activate

pip install insightface==0.7.3 modelscope==1.17.1 addict==2.4.0 datasets==2.21.0 oss2==2.18.6 simplejson==3.19.3 sortedcontainers==2.4.0 gfpgan==1.3.8 --require-virtualenv

cat <<EOF
pip done...


Now add this to settings/powerup.json

  "Faceswap": {
    "type": "faceswap"
  }

Download models to models/faceswap/
https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
...and inswapper_128.onnx from where you can find it
EOF
