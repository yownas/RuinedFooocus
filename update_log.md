### 1.20.0
* Added wildcard helper, just start typing __ and your wildcards will display

### 1.19.5
* Fixed old bug of switching models when the stop button is pressed (old code from OG-F)

### 1.19.4
* Old experimental lcm-pipeline removed
* Generate forever if Image Number is set to 0
* Updated comfy version to latest
* Nested wildcards now supported

### 1.19.3
* Random styles now correctly applying to each image

### 1.19.2
* Gradio Rollback to v3 until v4 is fixed

### 1.19.1
* WildCard Fixes
* Automatcially downloads LCM Models
* Now checks subdirectories for models

### 1.19.0
* Gradio update to V4

### 1.18.0
* New Random Style Selection
* Adding one button prompt overrides in wildcards now
* Added wildcards official
* Other stuff i'll have to get arljen to explain

### 1.17.2
* Limit seeds to 32 bits
* Sort model dropdowns
* Use caches better

### 1.17.1
* removed groop and faceswap as it was causing dependency issues on some systems

### 1.17.0
* Changed minimum cfg scale to 0
* Updated to latest comfy and diffusers (Now supports LCM Loras)
* You NEEED to set the custom settings to use lcm and sgm_sampler, steps of 4 and REALLY low config (between 0 and 1.5)

### 1.16.0
* Facewapping
* Groop

### 1.15.1
* Updated Comfy Version

### 1.15.0
* Different pipelines supporting lcm and sdxl/ssd
* Let async_worker handle model preload
* Lots of small fixes
* fixed metadata bug when stopped

### 1.14.1
* Fixed small issue with metadata not updating

### 1.14.0
* Added Metadata Viewer for Gallery items (Viewable in `Info` Tab)
* Refresh Files now also reloads your `styles.csv` file

### 1.13.0
* Automatically download 4xUltrasharp Upscaler
* Added the ability to upscale images wth upscaler of your choosing
* Changed Powerup Settings so if there is a missing key from defaults it adds it to your custome settings.

### 1.12.1
* Refactored backend code to allow for future pipeline changes

### 1.12.0
* Automatically read triggerwords from <lora_filename>.txt

### 1.11.0
* Updated Comfy Version
* Added support for [SSD-1B Models](https://huggingface.co/segmind/SSD-1B)

### 1.9.0
* Removed ref redundant code.

### 1.8.2
* Update Comfy version and fix changes :D

### 1.8.1
* Improved image2image and allowed settings to be changed when "custom" is selected form the PowerUp Tab.

### 1.8.0
* Added the basics for image 2 image
* Renamed Controlnet to PowerUp
* Now uses `powerup.json` as default

### 1.7.2
* Wildcards can now use subdirectories
* Fixed issue where if you placed 2 placeholders with the same name, you got the same results, a new one is now chosen
* Updated status to show model loading / vae decoding

### 1.7.1
* Update to one button prompt (provided by [Alrjen](https://github.com/AIrjen/OneButtonPrompt))

### 1.7.0
* Custom Controlnet Modes
* minor bugfixes
* moved the controlnet tab to its own ui file.

### 1.6.1
* Added sketch controlnet!

### 1.6.0
* Updated gradio version
* Added recolour controlnet!

### 1.5.2
* Restored gallery preview on all images
* renamed more variables to make sense
* bugfixes

### 1.5.1
* Added all the settings/customization to their own `settings` folder **NOTE:** you will need to move / copy your settings into the new directory
* Bugfix where clicking stop mid-generation stopped working
* code cleanup and some renaming
* inference model speed up
* now only shows gallery when needed

### 1.5.0
* removed metadata toggle, it will now always save metadata
* save your own custom performances
* tidied ui
* fix crash when failing to load loras
* hide all but one lora dropdown showing "None"

### 1.4.2
* change fooocusversion.py to version.py for easier updating
* Moved controlnet to its own tab for easier updates
* updated gradio version
* minor wording changes

### 1.4.1
* `paths.json` will now be updated if there are any missing defaults paths

### 1.4.0
* Now supports controlnet

### 1.3.0
* Updated onebutton prompt so you can now add multiple random prompts by clicking the `Add To Prompt` button

### 1.2.2
* Update comfy version - Lora weights are now calculated on the gpu so should apply faster
### 1.2.1
* Bug fixes and backend updates
* changed `resolutions.csv` to `resolutions.json`
* updated readme

### 1.2.0
* Prompt now splits correctly using `---`
* added the ability to change styles in the prompt by using <style:stylename>

### 1.1.7
* Added init image

### 1.1.6
* Fixed issue with wildcards if file not found.

### 1.1.5
* Fixed sorting on subfolders, so directories are displayed first

### 1.1.4
* Allowed main image window to recieve drag and drop
* Added a gallery preview underneath that will activate image window.

### 1.1.3
* Added support for subdirectories with models/loras so you can get all organised!

### 1.1.2
* showed imported image in gallery 
* moved `---` split into prompt generation
* correctly updates progressbar
* fixed importing of width / height

### 1.1.1
*  In the json prompt, setting a seed of `-1` allows you to generate a random seed

### 1.1.0
*  Render different subjects so you can process a whole list of prompts. Seperate each prompt by placing `---` on a new line

### 1.0.0
* New Beginnings. The official start of the updates!
