# RuinedFooocus

<img src="https://raw.githubusercontent.com/runew0lf/pmmconfigs/main/RuinedFooocus_ss.png" width=100%>

# The Future of AI Art is Here - Introducing RuinedFooocus

Forget everything you thought you knew about AI art generation - **RuinedFooocus** is here to completely reinvent the game! 

This groundbreaking new image creator combines the best aspects of **Stable Diffusion** and **Midjourney** into one seamless, cutting-edge experience. The days of messy installations and manual tweaking are over. With **RuinedFooocus**, stunning images spring to life with just a few words - no technical skills required.

Built from the DNA of its predecessors but enhanced with next-level AI, **RuinedFooocus** generates jaw-dropping visuals in seconds. Just type a prompt and watch in awe as your wildest creative visions are instantly brought to life. The possibilities are truly endless.

But the magic doesn't stop there. **RuinedFooocus** puts the power in your hands like never before. You're in complete control of the experience from start to finish. And with its streamlined interface and intuitive controls, anyone can create like a pro.

Don't waste another minute stuck in the past. The future is calling, and it's time to answer with **RuinedFooocus** - the new standard in AI artistry. Install it today and unlock your creative potential like never before.  

### The future of art starts now!

## Download

### Windows

You can directly download Fooocus with:

**[>>> Click here to download <<<](https://github.com/runew0lf/RuinedFooocus/releases/download/Release/RuinedFooocus_win641-0-0.7z)**

After you download the file, please uncompress it, and then run the "run.bat".

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/c49269c4-c274-4893-b368-047c401cc58c)

In the first time you launch the software, it will automatically download models:

1. It will download [sd_xl_base_1.0_0.9vae.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors) as the file "Fooocus\models\checkpoints\sd_xl_base_1.0_0.9vae.safetensors".
2. It will download [sd_xl_refiner_1.0_0.9vae.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors) as the file "Fooocus\models\checkpoints\sd_xl_refiner_1.0_0.9vae.safetensors".

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/d386f817-4bd7-490c-ad89-c1e228c23447)

If you already have these files, you can copy them to the above locations to speed up installation.

Note that if you see **"MetadataIncompleteBuffer"**, then your model files are corrupted. Please download models again.

### Linux

The command lines are

    git clone https://github.com/runew0lf/RuinedFooocus.git
    cd Fooocus
    conda env create -f environment.yaml
    conda activate fooocus
    pip install -r requirements_versions.txt

Then download the models: download [sd_xl_base_1.0_0.9vae.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors) as the file "Fooocus\models\checkpoints\sd_xl_base_1.0_0.9vae.safetensors", and download [sd_xl_refiner_1.0_0.9vae.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors) as the file "Fooocus\models\checkpoints\sd_xl_refiner_1.0_0.9vae.safetensors". **Or let Fooocus automatically download the models** using the launcher:

    python launch.py

Or if you want to open a remote port, use

    python launch.py --listen

### Mac/Windows(AMD GPUs)

Coming soon ...

### Colab

(Last tested - 2023 Sep 07)

| Colab | Info
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/runew0lf/RuinedFooocus/blob/main/colab.ipynb) | RuinedFooocus Colab

Note that sometimes this Colab will say like "you must restart the runtime in order to use newly installed XX". This can be safely ignored.

## Ruined Edition Features

1. Supports custom styles in `styles.csv`
2. Changed Resolutions and Styles to be in a dropdown instead of radio buttons
3. Apply multiple styles to one prompt and a send style to prompt button.
4. Ability to save full metadata for generated images embedded in PNG.
5. Ability to change default values of UI settings (loaded from `settings.json`).
6. Generate a completely random prompt (taken from onebuttonprompt) with its own "special" tab
7. Made Resolutions mode readable
8. Added theme support in `settings.json` you can see available themes [HERE](https://huggingface.co/spaces/gradio/theme-gallery)
9. Added `resolutions.csv` to allow users to add their own custom resolutions
10. Wildcards are now supported place see `wildcards\colors.txt` for an example. In your prompt make sure you type `__<filename>__` to activate ie `shiny new __colors__ Chevrolet pickup truck with big wheels`
11. If the option `--nobrowser` is passed the web browser won't automatically launch
12. Added Custom paths in `paths.json` to point to chekcpoints / loras and outputs director (**Note:** for windows paths either use `/` or `\\` instead of `\`)
13. Added support for custom Performance - enables samplers/scheduler, steps, refiner steps, cfg & clip skip (Check advanced tab)
14. Displays time taken for each render in the console. If `notification.mp3` exists in the root directory, this will play when the generations are complete.
15. Added **Cancel** button to stop generation. Thanks to [Yownas](https://github.com/yownas) and [MoonRide](https://github.com/MoonRide303/)
16. Pressing `Ctrl-Enter` is the same as pressing the generate button!

## Thanks

This codebase is a fork from the original amazing work by [lllyasviel](https://github.com/lllyasviel/Fooocus/discussions/117)
The wonderful [MoonRide](https://github.com/MoonRide303/) who also maintains an amazing fork!
The codebase starts from an odd mixture of [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and [ComfyUI](https://github.com/comfyanonymous/ComfyUI). (And they both use GPL license.)

## Discord Support Server
You can join our discord support server [Here](https://discord.gg/CvpAFya9Rr)

## Update Log
The log is [here](update_log.md).
