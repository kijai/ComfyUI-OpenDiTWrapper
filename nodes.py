import torch
import os
import sys
import gc
script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

from opendit.core.pab_mgr import set_pab_manager
#from opendit.core.parallel_mgr import enable_sequence_parallel, set_parallel_manager
from opendit.models.opensora import RFLOW, OpenSoraVAE_V1_2, STDiT3_XL_2, T5Encoder, text_preprocessing
from opendit.models.opensora.inference_utils import (
    append_score_to_prompts,
    extract_prompts_loop,
    merge_prompt,
    prepare_multi_resolution_info,
    split_prompt,
)

import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file
import folder_paths

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
    print("Flash Attention is available")
except:
    FLASH_ATTN_AVAILABLE = False
    print("WARNING! Flash Attention is not available, using much slower torch SDP attention")

class DownloadAndLoadOpenSoraModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [ 
                    'hpcai-tech/OpenSora-STDiT-v3'
                    ],
                   ),
            "precision": (['fp16','bf16','fp32'],
                    {
                    "default": 'bf16'
                    }),
            },
        }

    RETURN_TYPES = ("OPENDITMODEL",)
    RETURN_NAMES = ("opendit_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "OpenDitWrapper"

    def loadmodel(self, model, precision):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_name = model.rsplit('/', 1)[-1]
        model_path = os.path.join(folder_paths.models_dir, "opensora", model_name)
        
        if not os.path.exists(model_path):
            print(f"Downloading OpenSora model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model,
                            ignore_patterns=['*ema*'],
                            local_dir=model_path,
                            local_dir_use_symlinks=False)
            
        if not hasattr(self, "model"):
            print("Loading STDiT...")
            self.model = (
            STDiT3_XL_2(
                from_pretrained=model_path,
                qk_norm=True,
                enable_flash_attn=FLASH_ATTN_AVAILABLE,
                enable_layernorm_kernel=True,
                #input_size=latent_size,
                in_channels=4,
                caption_channels=4096,
                model_max_length=300
                ).to(offload_device, dtype).eval()
            )

        mm.soft_empty_cache()
        
        opendit_model = {
            'model': self.model, 
            'dtype': dtype
            }

        return (opendit_model,)
    
class DownloadAndLoadOpenSoraVAE:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [ 
                    'hpcai-tech/OpenSora-VAE-v1.2'
                    ],
                   ),
            "precision": (['fp16','bf16','fp32'],
                    {
                    "default": 'bf16'
                    }),
            },
        }

    RETURN_TYPES = ("OPENDITVAE",)
    RETURN_NAMES = ("opendit_vae",)
    FUNCTION = "loadmodel"
    CATEGORY = "OpenDitWrapper"

    def loadmodel(self, model, precision):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_name = model.rsplit('/', 1)[-1]
        model_path = os.path.join(folder_paths.models_dir, "opensora", model_name)
        
        if not os.path.exists(model_path):
            print(f"Downloading OpenSora model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model,
                            ignore_patterns=['*ema*'],
                            local_dir=model_path,
                            local_dir_use_symlinks=False)
            
        if not hasattr(self, "vae"):
            print("Loading VAE...")
            self.vae = (
            OpenSoraVAE_V1_2(
                from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
                micro_frame_size=17,
                micro_batch_size=4,
                ).to(offload_device, dtype).eval()
            )

        mm.soft_empty_cache()
        
        opendit_model = {
            'model': self.vae, 
            'dtype': dtype
            }

        return (opendit_model,)

class DownloadAndLoadOpenDiTT5Model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [ 
                    'city96/t5-v1_1-xxl-encoder-bf16'
                    ],
                   ),
            "precision": (['fp16','bf16','fp32'],
                    {
                    "default": 'bf16'
                    }),
            },
        }

    RETURN_TYPES = ("OPENDITT5",)
    RETURN_NAMES = ("opendit_t5_encoder",)
    FUNCTION = "loadmodel"
    CATEGORY = "OpenDitWrapper"

    def loadmodel(self, model, precision):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_name = model.rsplit('/', 1)[-1]
        model_path = os.path.join(folder_paths.models_dir, "t5", model_name)
        
        if not os.path.exists(model_path):
            print(f"Downloading OpenSora model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model,
                            ignore_patterns=['*ema*'],
                            local_dir=model_path,
                            local_dir_use_symlinks=False)
        
       
        if not hasattr(self, "text_encoder"):
            print("Loading Text Encoder...")
            self.text_encoder = T5Encoder(
            from_pretrained=model_path, model_max_length=300, device=device, dtype=dtype, shardformer=False
            )
            
        mm.soft_empty_cache()

        return (self.text_encoder,)
    
class OpenDiTConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            "opendit_t5_encoder": ("OPENDITT5",),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "camera_prompt": ("STRING", {"default": "", "multiline": True}),
            "aesthetic_score": ("FLOAT", {"default": 6.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            "flow_score": ("FLOAT", {"default": 0.0}, {"min": 0.0, "max": 100.0, "step": 0.1}),
            "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("OPENDITCOND",)
    RETURN_NAMES =("opendit_cond",)
    FUNCTION = "process"
    CATEGORY = "OpenDiTWrapper"

    def process(self, opendit_t5_encoder, prompt, camera_prompt, aesthetic_score, flow_score, keep_model_loaded=False):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        self.text_encoder = opendit_t5_encoder

        print("process prompt step by step...")
        # == process prompt step by step ==
        # 0. split prompt
        prompt_segment_list, loop_idx_list = split_prompt(prompt)

        # 1. append score
        prompt_segment_list = append_score_to_prompts(
            prompt_segment_list,
            aes=aesthetic_score if aesthetic_score > 0 else None,
            flow=flow_score if flow_score > 0 else None,
            camera_motion=camera_prompt if camera_prompt != "" else None,
        )

        # 2. clean prompt with T5
        prompt_segment_list = [text_preprocessing(prompt) for prompt in prompt_segment_list]

        # 3. merge to obtain the final prompt
        final_prompt = merge_prompt(prompt_segment_list, loop_idx_list)
        final_prompt_loop = extract_prompts_loop([final_prompt], 0) 
        print("final_prompt_loop: ", final_prompt_loop)

        self.text_encoder.t5.model.to(device)
        encoded_prompt = self.text_encoder.encode(final_prompt_loop)
        if not keep_model_loaded:
            self.text_encoder.t5.model.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        return (encoded_prompt,)   
class OpenDiTSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            "opendit_model": ("OPENDITMODEL",),
            "opendit_vae": ("OPENDITVAE",),
            "opendit_cond": ("OPENDITCOND",),
            "num_frames": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
            "width": ("INT", {"default": 426, "min": 1, "max": 2048, "step": 1}),
            "height": ("INT", {"default": 240, "min": 1, "max": 2048, "step": 1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
            "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "fps": ("INT", {"default": 24, "min": 1, "max": 60, "step": 1}),
            "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "OpenDiTWrapper"

    def process(self, opendit_model, opendit_vae, opendit_cond, num_frames, width, height, seed, steps, cfg, fps, keep_model_loaded=False):
        device = mm.get_torch_device()
        dtype = opendit_model['dtype']
        offload_device = mm.unet_offload_device()
        self.model = opendit_model['model']
        self.vae = opendit_vae['model']

        set_pab_manager(
            steps=steps,
            cross_broadcast=True,
            cross_threshold=[540, 940],
            cross_gap=6,
            spatial_broadcast=True,
            spatial_threshold=[540, 940],
            spatial_gap=2,
            temporal_broadcast=True,
            temporal_threshold=[540, 940],
            temporal_gap=4,
            #diffusion_skip=6
            #diffusion_skip_timestep= [1,1,1,0,0,0,0,0,0,0]
        )
        
        image_size = (height, width)
        input_size = (num_frames, *image_size)
        latent_size = self.vae.get_latent_size(input_size)

        scheduler = RFLOW(use_timestep_transform=True, num_sampling_steps=steps, cfg_scale=cfg)

        print("Sampling...")
        # == sampling ==
        torch.manual_seed(seed)
        z = torch.randn(1, self.vae.out_channels, *latent_size, device=device, dtype=dtype)

        mm.soft_empty_cache()
        gc.collect()

        multi_resolution = "STDiT2"
        additional_args = prepare_multi_resolution_info(
            multi_resolution, 1, image_size, num_frames, fps, device, dtype
        )
        print("additional_args: ", additional_args)
        final_cond = opendit_cond.copy()
        final_cond.update(additional_args)

        self.model.to(device)

        y_null = self.model.y_embedder.y_embedding[None].repeat(1, 1, 1)[:, None]
        final_cond["y"] = torch.cat([final_cond["y"], y_null], 0)

        samples = scheduler.sample(
            self.model,
            final_cond,
            z=z,
            device=device,
            progress=True,
            additional_args=additional_args,
            # mask=masks,  # Adjust or omit based on your needs
        )
        if not keep_model_loaded:
            self.model.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        self.vae.to(device)
        samples = self.vae.decode(samples.to(dtype), num_frames=num_frames)
        self.vae.to(offload_device)

        samples = samples.squeeze(0).permute(1, 2, 3, 0).float().cpu()
        normalized_tensor = torch.clamp(samples, -1, 1)
        
        tensor_min = normalized_tensor.min()
        tensor_max = normalized_tensor.max()
        normalized_tensor = (samples - tensor_min) / (tensor_max - tensor_min)
        normalized_tensor = torch.clamp(normalized_tensor, 0, 1)

        return (normalized_tensor,)   
     
NODE_CLASS_MAPPINGS = {
    "OpenDiTSampler": OpenDiTSampler,
    "OpenDiTConditioning": OpenDiTConditioning,
    "DownloadAndLoadOpenSoraModel": DownloadAndLoadOpenSoraModel,
    "DownloadAndLoadOpenSoraVAE": DownloadAndLoadOpenSoraVAE,
    "DownloadAndLoadOpenDiTT5Model": DownloadAndLoadOpenDiTT5Model
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenDiTSampler": "OpenDiT Sampler",
    "OpenDiTConditioning": "OpenDiT Conditioning",
    "DownloadAndLoadOpenSoraModel": "(Down)Load OpenSora Model",
    "DownloadAndLoadOpenSoraVAE": "(Down)Load OpenSora VAE",
    "DownloadAndLoadOpenDiTT5Model": "(Down)Load OpenDiT T5 Model"
}