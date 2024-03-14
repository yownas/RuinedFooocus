### 1.34.0
* Added llyasviel's prompt tokenizer model for improving prompts, This can be used by adding the `Flufferizer` style

### 1.33.1
* Moved upscale to its own pipeline
* Various Bug Fixes

### 1.33.0
* Updated to latest comfy version
* Now supports Playground 2.5 model!

### 1.32.0
* Various Bugfixes
* Adds the ability to upgrade to a newer torch and xformers, just create a blank `reinstall` file in the RF directory

### 1.31.0
* Added presets / custom preset for OBP
* Various Bugfixes

### 1.30.0
* The OBP Update
* added "the tokinator" mode
* Fintetunes on autonegative prompt: -- Landscape removed
* Added list of artist shorthands (like van gogh)
* Added secret sauce mode (mixing OBP + SDXL styles). Happens more on lower insanity levels
* Added more consistency on lower insanity levels
* EVO with OPB variant works better:
* Small prompts now work
* (almost) always changes something
* Fix for line ends creeping into EVO + OPB Variant
* Added loads of new styles into the lists
* Added loads of more fictional characters into the lists (finally, harley quinn is in :p)

### 1.29.0
* Automatic Negative prompt, save yourself the heartache and hassle of writing negative prompts!

### 1.28.2
* Fixed some dodgy regex to support wildcards with `-` in the name

### 1.28.1
* Added better error descriptions for the civit api

### 1.28.0
* Automatically grabs lora keywords from civit on startup

### 1.27.0
* Advance prompt editing - See [Wiki](https://github.com/runew0lf/RuinedFooocus/wiki/Features#advanced-prompt-editing)
* Moved Evolve to the `Powerup` tab to be cleaner
* Bufixes

### 1.26.1
* Changed so loras with triggerwords textfiles dont automatically get added, instead displays trigger words to add manually

### 1.26.0
* Fixed issue with controlnet caching
* Custom resolutions can now be saved directly, just select custom from the resolution dropdown (Lavorther)
* re-enabled comfy messages

### 1.25.5
* Make Custom option for Performance Mode copy last selected mode's setings (Lavorther)

### 1.25.4
* Fixed an issue with wildcards and strengths getting stuck in Catastrophic backtracking

### 1.25.3
* Changed a lot of os.path to pathlibs
* Worked on backend code for future extensibility
* Added support for --auth

### 1.25.2
* Fixed an os.path issue by replaceing with pathlib

### 1.25.1
* Now displays 🗒️ at the end of the loraname if it has a keyword .txt file

### 1.25.0
* Prompt Bracketing Now works ie `[cat|dog]`
* Updated comfy version

### 1.24.2
* More Tidying and tweaks
* Autoselect LCM lora if Lcm is selected from the quality or sampler dropdowns

### 1.24.1
* Code Tidying and tweaks

### 1.24.0
* Added Evolve which generates variations on your generated picture.
* Fixed wild imports
* Added instant One Button Prompt
* Added keybind of `ctrl+shift` to toggle `hurt me plenty`` mode
* Code Cleanup
* Fixed Issue with dropdown box's being case sensitive when sorting

### 1.23.2
* Adds token count (thanks Lavorther)
* Fixed lora keywords only working the first time the lora is loaded

### 1.23.1
* Use OneButtonPrompt with infinite mode, simply put a tick in `BYPASS SAFETY PROTOCOLS`

### 1.23.0
* Inpainting Update: Adds inpainting to the powerup tab

### 1.22.0
* Now comes with a built in clip interrogator, just drag your image onto the main image to generate the prompt

### 1.21.0
* Π Time - Click the small Π symbol to get a fullscreen image of your last generation great for slideshows
* updated comfy version, NEW SAMPLER TIME

### 1.20.0
* Added wildcard helper, just start typing __ and your wildcards will display
* Added slideshow

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
