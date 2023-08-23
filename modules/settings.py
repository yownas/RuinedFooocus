from os.path import exists
import json
import modules


def load_settings():
    settings = {}
    settings["advanced_mode"] = False
    settings["image_number"] = 1
    settings["seed_random"] = True
    settings["seed"] = 0
    settings["style"] = "Style: sai-cinematic"
    settings["prompt"] = ""
    settings["negative_prompt"] = ""
    settings["performance"] = "Speed"
    settings["resolution"] = "1152x896 (4:3)"
    settings["sharpness"] = 2.0
    settings["img2img_mode"] = False
    settings["img2img_start_step"] = 0.06
    settings["img2img_denoise"] = 0.94
    settings["base_model"] = modules.path.default_base_model_name
    settings["refiner_model"] = modules.path.default_refiner_model_name
    settings["lora_1_model"] = modules.path.default_lora_name
    settings["lora_1_weight"] = modules.path.default_lora_weight
    settings["lora_2_model"] = "None"
    settings["lora_2_weight"] = modules.path.default_lora_weight
    settings["lora_3_model"] = "None"
    settings["lora_3_weight"] = modules.path.default_lora_weight
    settings["lora_4_model"] = "None"
    settings["lora_4_weight"] = modules.path.default_lora_weight
    settings["lora_5_model"] = "None"
    settings["lora_5_weight"] = modules.path.default_lora_weight
    settings["save_metadata"] = True
    settings["theme"] = "None"

    if exists("settings.json"):
        with open("settings.json") as settings_file:
            try:
                settings_obj = json.load(settings_file)
                for k in settings.keys():
                    if k in settings_obj:
                        settings[k] = settings_obj[k]
            except Exception as e:
                print(e)
                pass
            finally:
                settings_file.close()

    return settings
