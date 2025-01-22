from qwen_vl_utils import process_vision_info
from PIL import Image

def preprocess_image_nearest(image_path: str, image_resolution: int, perform_resize: bool = False) -> tuple[int, int]:
    try:
        with Image.open(image_path) as image:
            original_width, original_height = image.width, image.height
            max_dimension = max(original_width, original_height)

            if max_dimension > image_resolution:
                resize_factor = image_resolution / max_dimension
                new_width = int(original_width * resize_factor)
                new_height = int(original_height * resize_factor)
            else:
                new_width, new_height = original_width, original_height

            if perform_resize:
                resized_image = image.resize((new_width, new_height), resample=Image.NEAREST)
                if image.mode != "RGB":
                    resized_image = resized_image.convert("RGB")
                return new_width, new_height, resized_image

    except Exception as e:
        raise IOError(f"Can not open the image {image_path}: {e}")

    return new_height, new_width



def bulid_message(prompt, config, eval_name, bev_mf_paths):
    """
    Build the message to send to the OpenAI API. If config specifies only BEV, the function
    will use only the BEV image. Otherwise, it defaults to using both if available.
    """
    # if eval_name in ['scanrefer', 'multi3dref'] and config.base_model:
    #     prompt += " Carefully read and understand the prompt's description of the object(s). Indicate with 'Yes' if there are objects that match the description, followed by their IDs in the format '<OBJ###>' (e.g., '<OBJ001>, <OBJ002>'). If there are multiple matching objects, separate them with commas. Respond with 'No' if no objects match the description."
    user_prompt = []

    for bev_mf_path in bev_mf_paths:
        train_height, train_width = preprocess_image_nearest(bev_mf_path, config.resolution)
        user_prompt.append({"type": "image", "image": bev_mf_path, "resized_height": train_height, "resized_width": train_width})

    user_prompt.append({"type": "text", "text": prompt})

    message = []
    message.append({"role": "user",  "content": user_prompt})
    return message

def get_response_qwen(model, processor, prompts, bev_mf_paths, config, eval_name):
            
    messages = []
    for prompt, bev_mf_path in zip(prompts, bev_mf_paths):
        messages.append(bulid_message(prompt, config, eval_name, bev_mf_path))

    # Preparation for inference
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Batch Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_texts