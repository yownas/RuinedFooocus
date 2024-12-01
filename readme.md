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

**[>>> Click here to download <<<](https://github.com/runew0lf/RuinedFooocus/releases/download/Release-1.25.2/RuinedFooocus_win64_1-25-2.7z)**

## NOTE YOU WILL NEED TO EXTRACT USING THE LATEST [7ZIP](https://www.7-zip.org/download.html)

After you download the file, please uncompress it, and then run the "run.bat".

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/c49269c4-c274-4893-b368-047c401cc58c)

In the first time you launch the software, it will automatically download models:

1. It will download [sd_xl_base_1.0_0.9vae.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors) as the file "Fooocus\models\checkpoints\sd_xl_base_1.0_0.9vae.safetensors".

If you already have these files, you can copy them to the above locations to speed up installation.

Note that if you see **"MetadataIncompleteBuffer"**, then your model files are corrupted. Please download models again.

### Linux

The command lines are

    git clone https://github.com/runew0lf/RuinedFooocus.git
    cd RuinedFooocus
    virtualenv venv
    source venv/bin/activate
    pip install -r requirements_versions.txt

Then download the models: download [sd_xl_base_1.0_0.9vae.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors) as the file "Fooocus\models\checkpoints\sd_xl_base_1.0_0.9vae.safetensors", and download [sd_xl_refiner_1.0_0.9vae.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors) as the file "Fooocus\models\checkpoints\sd_xl_refiner_1.0_0.9vae.safetensors". **Or let Fooocus automatically download the models** using the launcher:

    python launch.py

Or if you want to open a remote port, use

    python launch.py --listen


## Ruined Edition Features

1. Supports custom styles in `settings/styles.csv`
2. Changed Resolutions and Styles to be in a dropdown instead of radio buttons
3. Apply multiple styles to one prompt and a send style to prompt button.
4. Ability to save full metadata for generated images embedded in PNG.
5. Ability to change default values of UI settings (loaded from `settings/settings.json`).
6. Generate a completely random prompt (taken from onebuttonprompt) with its own "special" tab (now officially supported by [Arljen](https://github.com/AIrjen/OneButtonPrompt))
7. Made Resolutions mode readable
8. Added theme support in `settings/settings.json` you can see available themes [HERE](https://huggingface.co/spaces/gradio/theme-gallery)
9. Added `settings/resolutions.json` to allow users to add their own custom resolutions
10. Wildcards are now supported place see `wildcards\colors.txt` for an example. In your prompt make sure you type `__<filename>__` to activate ie `shiny new __colors__ Chevrolet pickup truck with big wheels`
11. If the option `--nobrowser` is passed the web browser won't automatically launch
12. Added Custom paths in `settings/paths.json` to point to chekcpoints / loras and outputs director (**Note:** for windows paths either use `/` or `\\` instead of `\`)
13. Added support for custom Performance - enables samplers/scheduler, steps, refiner steps, cfg & clip skip (Check advanced tab)
14. Displays time taken for each render in the console. If `notification.mp3` exists in the root directory, this will play when the generations are complete.
15. Added **Cancel** button to stop generation. Thanks to [Yownas](https://github.com/yownas) and [MoonRide](https://github.com/MoonRide303/)
16. Pressing `Ctrl-Enter` is the same as pressing the generate button!
17. Adds the ability to add loras to just prompt ala auto111 style eg `<lora:MyAwesomeLora-SL:0.90>`
18. Drag and drop previous **Ruined** images to the main window, it will autofill the prompt with the generation metadata.
19. The ability to post json metadata into the prompt window so you dont have to go into ANY settings tabs when you click generate. (**Note:** setting a seed of `-1` will generate a random seed)
20. Render different subjects so you can process a whole list of prompts. Seperate each prompt by placing `---` on a new line
21. Supports subdirectories for models/loras so you can get all organised!
22. Add <style:stylename> to prompt to set the required style
23. Controlnet! Check out the **PowerUp** tab, once selected you'll be able to upload your controlnet base image into a new special area in the tab
24. Save your own custom perfomances and easily load them back in with the dropdown
25. Create your own default ControlNet settings, or just get into the hardcore details with the custom **PowerUp** mode
26. Image2Image mode is now active and available in the **PowerUp** tab
27. Added Support for [SSD-1B Models](https://huggingface.co/segmind/SSD-1B)
28. Automatically read triggerwords from <lora_filename>.txt just place a `.txt` file of the same name (minus the .safetensors) and it will read the triggerwords from the file
29. Upscaling using whatever upscaler you prefer `4xUltrasharp` by default. Look for the new option in the powerup tab
30. Metadata Viewer now available in the `Info` tab
31. Now supports the [SDXL LCM LORA](https://huggingface.co/latent-consistency/lcm-lora-sdxl/tree/main)
32. Generate Forever if Image Number is set to 0.
33. Clip Interrogator, just drag your image onto the main image to generate the prompt
34. Inpainting, Available in the `PowerUp` tab, simple check the box and it will either take a new image or the selected image in your gallery
35. Evolve, takes your current generation and evolves it into different variations, best used with a fixed seed!
36. Support --auth=username/password for rudimentary security (Forced when using --share)
37. Automatically downloads your Lora triggerwords from civit and displays them for you
38. Automatic Negative prompt, save yourself the heartache and hassle of writing negative prompts!
39. If the file `reinstall` exists upgrade xformers and torch to 2.1.2 (to upgrade simply create a blank file called `reinstall`)
40. Support [LayerDiffuse](https://github.com/huchenlei/ComfyUI-layerdiffuse) PowerUp.
41. New model/LoRA select with support for animated thumbnails.
42. SD3 Support, make sure to use `sd3_medium_incl_clips.safetensors` - Note: *If the pictures look "strange" change to euler and simple*
43. Use [merge-files](https://github.com/runew0lf/RuinedFooocus/wiki/Checkpoint-merges) as checkpoints.
44. MergeMaker in Models, where you can create your own merge-files/checkpoints
45. Experimental support for diffusers and models from Huggingface
46. RemBG PowerUp, remove background of images
47. Img2STL PowerUp, convert images to .stl files for 3D printers
48. Flux support.
49. Support for GGUF Flux files and safetensors containing only Unet.
50. Florence for image Interrogation
51. Llama 3.2
52. Support "BaseModel"

## Thanks

This codebase is a fork from the original amazing work by [lllyasviel](https://github.com/lllyasviel/Fooocus/discussions/117)
The wonderful [MoonRide](https://github.com/MoonRide303/) who also maintains an amazing fork!
The codebase starts from an odd mixture of [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and [ComfyUI](https://github.com/comfyanonymous/ComfyUI). (And they both use GPL license.)

## Discord Support Server
You can join our discord support server [Here](https://discord.gg/CvpAFya9Rr)

## Update Log
The log is [here](update_log.md).
