@echo off
echo Some systems fail to install the insightface module, which is why this is not
echo part of the default install.
pause
echo Please wait...
..\..\python_embeded\python.exe -m pip install insightface==0.7.3 gfpgan==1.3.8 git+https://github.com/rodjjo/filterpy.git
echo.
echo Now add this to your RuinedFooocus\settings\powerup.json
echo.
echo "Faceswap": {
echo    "type": "faceswap"
echo }
echo.
echo Download the required models to models\faceswap\:
echo.
echo https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
echo and inswapper_128.onnx from where you can find it
echo.
echo You can find more info on the RuinedFooocus Github or Discord server.
echo.
pause
