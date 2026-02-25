import base64
import copy
from io import BytesIO
from typing import List

import decord
import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import extract_vision_info
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.models.simple import qwen2_5_vl
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL

from spa3_vlm.data.utils import load_and_preprocess_images
from spa3_vlm.model.spa3_vlm import Spa3_VLMForConditionalGeneration


class Spa3_VLM(Qwen2_5_VL):

    def __init__(self, *args, **kwargs):
        qwen2_5_vl.Qwen2_5_VLForConditionalGeneration = Spa3_VLMForConditionalGeneration
        super().__init__(*args, **kwargs)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="Model Responding"
        )
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}"
                    )

            messages = []
            processed_visuals = []
            for i, context in enumerate(contexts):

                message = [
                    {"role": "system", "content": "You are a helpful assistant."}
                ]

                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith(
                        (".mp4", ".avi", ".mov")
                    ):  # Video file
                        vr = decord.VideoReader(visual)
                        image_num = len(vr)
                        # sample max_num_frames frame indices from the video
                        if image_num < self.max_num_frames:
                            frame_indices = np.arange(image_num)
                        else:
                            frame_indices = np.linspace(
                                0, image_num - 1, self.max_num_frames
                            ).astype(int)
                        # read the frames
                        frames = [vr[i].asnumpy() for i in frame_indices]
                        visual_content = []
                        for frame in frames:
                            image = Image.fromarray(frame).convert("RGB")
                            visual_content.append({"type": "image", "image": image})
                        message.append({
                            "role": "user",
                            "content": visual_content + [{
                                "type": "text", "text": context
                            }],
                        })

                    elif isinstance(visual, Image.Image):  # Single image
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        message.append({
                            "role": "user",
                            "content": [{
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{base64_string}",
                            }, {
                                "type": "text", "text": context
                            }],
                        })
                    elif isinstance(visual, (list, tuple)) and all(
                        isinstance(v, Image.Image) for v in visual
                    ):  # Multiple images
                        image_content = []
                        image_count = 0
                        for v in visual:
                            base64_image = v.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            if self.add_frame_index:
                                image_content.append({
                                    "type": "text",
                                    "text": "Frame-{}: ".format(image_count),
                                })
                            image_content.append({
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{base64_string}",
                            })
                            image_count += 1
                        message.append({
                            "role": "user",
                            "content": image_content + [{
                                "type": "text", "text": context
                            }],
                        })
                    else:
                        message.append({
                            "role": "user",
                            "content": [{"type": "text", "text": context}],
                        })
                else:
                    message.append(
                        {"role": "user", "content": [{"type": "text", "text": context}]}
                    )

                messages.append(message)

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # image_inputs, video_inputs = process_vision_info(messages)

            geometry_encoder_inputs = []
            image_inputs = []
            patch_size = self.processor.image_processor.patch_size
            merge_size = self.processor.image_processor.merge_size
            for message in messages:
                vision_info = extract_vision_info(message)
                cur_geometry_encoder_inputs = []
                for ele in vision_info:
                    if "image" in ele:
                        image = ele["image"]
                        if isinstance(image, Image.Image):
                            pass
                        elif isinstance(image, str) and "base64," in image:
                            _, base64_data = image.split("base64,", 1)
                            data = base64.b64decode(base64_data)
                            # fix memory leak issue while using BytesIO
                            with BytesIO(data) as bio:
                                image = copy.deepcopy(Image.open(bio))
                        else:
                            raise NotImplementedError("Unsupported image type")

                    else:
                        raise NotImplementedError("Unsupported vision info type")

                    assert isinstance(
                        image, Image.Image
                    ), f"Unsupported image type: {type(image)}"
                    image = load_and_preprocess_images([image])[0]
                    cur_geometry_encoder_inputs.append(copy.deepcopy(image))
                    _, height, width = image.shape
                    # merge_size = 2
                    if (width // patch_size) % merge_size > 0:
                        width = width - (width // patch_size) % merge_size * patch_size
                    if (height // patch_size) % merge_size > 0:
                        height = (
                            height - (height // patch_size) % merge_size * patch_size
                        )
                    image = image[:, :height, :width]
                    image_inputs.append(image)

                geometry_encoder_inputs.append(torch.stack(cur_geometry_encoder_inputs))
            inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
                do_rescale=False,
            )
            device = "cuda" if self.device_map == "auto" else self.device
            if getattr(self.model.config, "use_geometry_encoder", False) or getattr(
                self.model.config, "use_vggt_feature", False
            ):
                inputs["geometry_encoder_inputs"] = [
                    feat.to(device) for feat in geometry_encoder_inputs
                ]
            inputs = inputs.to(device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, cont)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for i, ans in enumerate(answers):
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), ans
                )
                pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
