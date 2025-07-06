import os
import re
import json
import numpy as np
from tqdm import tqdm
import argparse

# Defines the mapping from action IDs to their text representations.
ACTION_MAPPING = {
    0: {"text": "D. Stop", "description": "Stop"},
    1: {"text": "A. Move forward 25 cm", "description": "Move forward 25 cm"},
    2: {"text": "B. Turn left 30 degrees", "description": "Turn left 30 degrees"},
    3: {"text": "C. Turn right 30 degrees", "description": "Turn right 30 degrees"}
}

def get_step_images(ep_folder):
    """Gets and sorts all step images from a trajectory directory."""
    files = os.listdir(ep_folder)
    return sorted(
        [f for f in files if re.match(r"step_(\d+)\.jpg", f)],
        key=lambda x: int(re.match(r"step_(\d+)\.jpg", x).group(1))
    )
 
def create_navigation_data(base_folder, max_frames=16, num_actions=4):
    """
    Creates the base navigation dataset.
    :param base_folder: The root directory containing all scene folders.
    :param max_frames: The maximum number of historical frames to consider per step.
    :param num_actions: The number of future actions to predict.
    :return: A dictionary containing the processed data.
    """
    ep_folders = [os.path.join(base_folder, f) for f in os.listdir(base_folder) 
                  if os.path.isdir(os.path.join(base_folder, f))]
    
    all_data = {}

    for ep_folder in tqdm(ep_folders, desc="Processing Scenes"):
        ep_name = os.path.basename(ep_folder)
        # Find the actual trajectory subfolder (e.g., traj_0)
        subfolder = next(
            (os.path.join(ep_folder, f) for f in os.listdir(ep_folder) 
            if os.path.isdir(os.path.join(ep_folder, f))),
            None
        )
        if not subfolder:
            print(f"⚠️  Skipping {ep_name}: No trajectory subfolder found.")
            continue

        try:
            with open(os.path.join(subfolder, "instruction.txt"), "r", encoding='utf-8') as f:
                instruction = f.read().strip()
            
            with open(os.path.join(subfolder, "action_list.json"), "r") as f:
                actions = json.load(f)

            step_images = get_step_images(subfolder)

            if len(actions) != len(step_images):
                print(f"❌ Data mismatch in {subfolder}: {len(actions)} actions vs {len(step_images)} images.")
                continue

            ep_steps = []
            for i, img in enumerate(step_images):
                # Sample historical frames evenly.
                hist_indices = np.linspace(0, i - 1, min(i, max_frames), dtype=int)
                
                current_step = int(re.match(r"step_(\d+)\.jpg", img).group(1))
                
                # Predict the next N actions.
                next_actions_ids = []
                for delta in range(1, num_actions + 1):
                    target_step = current_step + delta
                    if target_step < len(actions):
                        next_actions_ids.append(actions[target_step])
                    else:
                        next_actions_ids.append(0)  # Pad with 'Stop' action if out of bounds.
                
                action_texts = [ACTION_MAPPING[action_id]["text"] for action_id in next_actions_ids]
                action_str = ", ".join(action_texts)

                prev_actions_text = [ACTION_MAPPING[actions[idx]]["text"] for idx in hist_indices]

                ep_steps.append({
                    "step_id": current_step,
                    "current_image": img,
                    "previous_images": [step_images[idx] for idx in hist_indices],
                    "next_actions": next_actions_ids,
                    "action_text": action_str,
                    "previous_actions": prev_actions_text
                })

            all_data[ep_name] = {
                "instruction": instruction,
                "traj_path": os.path.basename(subfolder),
                "total_steps": len(ep_steps),
                "steps": ep_steps
            }

        except FileNotFoundError as e:
            print(f"❌ Missing file in {subfolder}: {e}")
        except Exception as e:
            print(f"❌ Failed to process {subfolder}: {str(e)}")

    return all_data

def add_step_prompts(navigation_data, num_actions=4):
    """
    Adds a prompt template to each step.
    :param navigation_data: Data returned from create_navigation_data.
    :param num_actions: The number of predicted actions, used for generating the prompt.
    :return: Data with prompts added.
    """
    option_list = "\n".join([ACTION_MAPPING[k]["text"] for k in sorted(ACTION_MAPPING.keys())])
    
    example_actions = [
        "A. Move forward 25 cm", "B. Turn left 30 degrees", 
        "C. Turn right 30 degrees", "D. Stop"
    ]
    # Cycle through example actions if num_actions > 4
    example_str = ", ".join([example_actions[i % len(example_actions)] for i in range(num_actions)])

    for ep_key in tqdm(navigation_data.keys(), desc="Generating Prompts"):
        episode = navigation_data[ep_key]
        instruction = episode["instruction"]
        
        for step in episode["steps"]:
            history_count = len(step["previous_images"])
            history_placeholders = " ".join(["<image>"] * history_count) if history_count > 0 else "None"
            
            action_word = "action" if num_actions == 1 else "actions"

            prompt_template = f"""
Imagine you are a robot designed for navigation tasks. Your instruction is {instruction!r}.
You are provided with:
- Historical observations: {history_placeholders} (if there are more than 16 frames, only 16 frames will be selected evenly from the total history)
- Current observation: <image>

Your task is to select the next {num_actions} {action_word} based on this information.
Options:
{option_list}

Answer the formatting requirements:
1. must contain {num_actions} action predictions
2. use full option text
3. be separated by commas
4. example: "{example_str}"
""".strip()

            step["prompt"] = prompt_template
    
    return navigation_data

def convert_to_multimodal(navigation_data, image_base_path):
    """
    Converts the processed data into a multimodal training format.
    :param navigation_data: Data returned from add_step_prompts.
    :param image_base_path: The root path for the image files.
    :return: A list formatted for multimodal model training.
    """
    converted_data = []
    
    for ep_id, episode in navigation_data.items():
        traj_path = os.path.join(ep_id, episode["traj_path"])
        
        for step in tqdm(episode["steps"], desc=f"Converting {ep_id}"):
            all_image_paths = [
                os.path.join(image_base_path, traj_path, img)
                for img in step["previous_images"]
            ] + [os.path.join(image_base_path, traj_path, step["current_image"])]
            
            conversation = {
                "conversations": [
                    {
                        "from": "human",
                        "value": step["prompt"]
                    },
                    {
                        "from": "gpt",
                        "value": step["action_text"]
                    }
                ],
                "images": all_image_paths
            }
            
            converted_data.append(conversation)
    
    return converted_data

def main_process(args):
    """
    The main data processing pipeline.
    """
    print("Step 1/3: Creating base navigation dataset...")
    nav_data = create_navigation_data(args.base_folder, args.max_frames, args.num_actions)
    
    print("\nStep 2/3: Adding prompt templates...")
    prompted_data = add_step_prompts(nav_data, args.num_actions)
    
    print("\nStep 3/3: Converting to multimodal format...")
    multimodal_data = convert_to_multimodal(prompted_data, args.image_base_path)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(multimodal_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Processing complete! Data saved to: {args.output_file}")
    print(f"Total records generated: {len(multimodal_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process VLN-Ego data for multimodal training.")
    
    parser.add_argument(
        '--base_folder', 
        type=str, 
        required=True,
        help='Path to the base directory containing scene folders (e.g., VLN-Ego/rgb_images_r2r_debug/).'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        required=True,
        help='Path to save the final JSON output file (e.g., VLN-Ego/r2r_16_images_4act_debug.json).'
    )
    parser.add_argument(
        '--image_base_path', 
        type=str, 
        required=True,
        help='The base path where the actual image files are stored (e.g., data/r2r_training_rgb/).'
    )
    parser.add_argument(
        '--max_frames', 
        type=int, 
        default=16,
        help='Maximum number of historical frames to consider for each step.'
    )
    parser.add_argument(
        '--num_actions', 
        type=int, 
        default=4,
        help='The number of future actions to predict at each step.'
    )
    
    args = parser.parse_args()
    main_process(args)
