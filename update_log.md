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
