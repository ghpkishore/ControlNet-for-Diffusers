# ControlNet-for-Any-Basemodel [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BI0TobTdjTI1VBSTjLXKOfh6Ps7uj6Ye?usp=sharing)

This repository provides the simplest tutorial code for developers using ControlNet with basemodel in the diffuser framework instead of WebUI. Our work builds highly on other excellent works. Although theses works have made some attemptes, there is no tutorial for supporting diverse ControlNet in diffusers.

<center><img src="https://github.com/lllyasviel/ControlNet/raw/main/github_page/he.png" width="100%" height="100%"></center> 

We have also supported [T2I-Adapter-for-Diffusers](https://github.com/haofanwang/T2I-Adapter-for-Diffusers), [Lora-for-Diffusers](https://github.com/haofanwang/Lora-for-Diffusers). Don't be mean to give us a star if it is helful to you.

# ControlNet + Anything-v3
Our goal is to replace the basemodel of ControlNet and infer in diffusers framework. The original [ControlNet](https://github.com/lllyasviel/ControlNet) is trained in pytorch_lightning, and the [released weights](https://huggingface.co/lllyasviel/ControlNet/tree/main/models) with only [stable-diffusion-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) as basemodel. However, it is more flexible for users to adopt their own basemodel instead of sd-1.5. Now, let's take [anything-v3](https://huggingface.co/Linaqruf/anything-v3.0/tree/main) as an example. We will show you how to achieve this (ControlNet-AnythingV3) step by step. We do provide a Colab demo [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BI0TobTdjTI1VBSTjLXKOfh6Ps7uj6Ye?usp=sharing), but it only works for Colab Pro users with larger RAM.

### (1) The first step is to replace basemodel. 

Fortunately, ControlNet has already provided a [guideline](https://github.com/lllyasviel/ControlNet/discussions/12) to transfer the ControlNet to any other community model. The logic behind is as below, where we keep the added control weights and only replace the basemodel. Note that this may not work always, as ControlNet may has some trainble weights in basemodel.
 
 ```bash
 NewBaseModel-ControlHint = NewBaseModel + OriginalBaseModel-ControlHint - OriginalBaseModel
 ```

First, we clone this repo from ControlNet.
 ```bash
 git clone https://github.com/lllyasviel/ControlNet.git
 cd ControlNet
 ```

Then, we have to prepared required weights for [OriginalBaseModel](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) (path_sd15), [OriginalBaseModel-ControlHint](https://huggingface.co/lllyasviel/ControlNet/tree/main/models) (path_sd15_with_control), [NewBaseModel](https://huggingface.co/Linaqruf/anything-v3.0/tree/main) (path_input). You only need to download following weights, and we use pose as ControlHint and anything-v3 as our new basemodel for instance. We put all weights inside ./models.

 ```bash
 path_sd15 = './models/v1-5-pruned.ckpt'
 path_sd15_with_control = './models/control_sd15_openpose.pth'
 path_input = './models/anything-v3-full.safetensors'
 path_output = './models/control_any3_openpose.pth'
 ```
 
 Finally, we can directly run
 ```bash
 python tool_transfer_control.py
 ```

If successful, you will get the new model. This model can already be used in ControlNet codebase.

```bash
models/control_any3_openpose.pth
 ```

If you want to try with other models, you can just define your own path_sd15_with_control and path_input.

### (2) The second step is to convert into diffusers

Gratefully, [Takuma Mori](https://github.com/takuma104) has supported it in this recent [PR](https://github.com/huggingface/diffusers/pull/2407), so that we can easily achieve this. As it is still under-devlopement, so it may be unstable, thus we have to use a specific commit version. I will reformat this section once the PR is mergered into diffusers.

```bash
git clone https://github.com/takuma104/diffusers.git
cd diffusers
git checkout 9a37409663a53f775fa380db332d37d7ea75c915
pip install .
```

Given the path of the generated model in step (1), run
```bash
python ./scripts/convert_controlnet_to_diffusers.py --checkpoint_path control_any3_openpose.pth  --dump_path control_any3_openpose --device cpu
```

We have the saved model in control_any3_openpose. Now we can test it as regularly.

```bash
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image

pose_image = load_image('https://huggingface.co/takuma104/controlnet_dev/resolve/main/pose.png')
pipe = StableDiffusionControlNetPipeline.from_pretrained("control_any3_openpose").to("cuda")

pipe.safety_checker = lambda images, clip_input: (images, False)

image = pipe(prompt="1gril,masterpiece,graden", controlnet_hint=pose_image).images[0]
image.save("generated.png")
```

The generated result may not be good enough as the pose is kind of hard. So to make sure everything goes well, we suggest to generate a normal pose via [PoseMaker](https://huggingface.co/spaces/jonigata/PoseMaker) or use our provided pose image in ./images/pose.png.

<img src="https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/images/pose.png" width="25%" height="25%"> <img src="https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/images/generated.png" width="25%" height="25%">


# ControlNet + Inpainting

This is to support ControlNet with the ability to only modify a target region instead of full image just like [stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting). For now, we provide the condition (pose, segmentation map) beforehands, but you can use adopt pre-trained detector used in ControlNet.

We have provided the [required pipeline](https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/pipeline_stable_diffusion_controlnet_inpaint.py) for usage. But please note that this file is fragile without complete testing, we will consider support it in diffusers framework formally later. Also, we find that ControlNet (sd1.5 based) is not compatible to [stable-diffusion-2-inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting), as some layers have different modules and dimension, if you forcibly load the weights and skip those unmatching layers, the result will be bad

```bash
# assume you already know the absolute path of installed diffusers
cp pipeline_stable_diffusion_controlnet_inpaint.py  PATH/pipelines/stable_diffusion
```

Then, you need to import this new added pipeline in corresponding files
```
PATH/pipelines/__init__.py
PATH/__init__.py
```

Now, we can run

```
import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline

# we have downloaded models locally, you can also load from huggingface
# control_sd15_seg is converted from control_sd15_seg.safetensors using instructions above
pipe_control = StableDiffusionControlNetInpaintPipeline.from_pretrained("./diffusers/control_sd15_seg",torch_dtype=torch.float16).to('cuda')
pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained("./diffusers/stable-diffusion-inpainting",torch_dtype=torch.float16).to('cuda')

# yes, we can directly replace the UNet
pipe_control.unet = pipe_inpaint.unet
pipe_control.unet.in_channels = 4

# we also the same example as stable-diffusion-inpainting
image = load_image("https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png")
mask = load_image("https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png")

# the segmentation result is generated from https://huggingface.co/spaces/hysts/ControlNet
control_image = load_image('tmptvkkr0tg.png')

image = pipe_control(prompt="Face of a yellow cat, high resolution, sitting on a park bench", 
                     negative_prompt="lowres, bad anatomy, worst quality, low quality",
                     controlnet_hint=control_image, 
                     image=image,
                     mask_image=mask,
                     num_inference_steps=100).images[0]

image.save("inpaint_seg.jpg")
```

The following images are original image, mask image, segmentation (control hint) and generated new image.

<img src="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png" width="20%" height="20%"> <img src="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png" width="20%" height="20%"> <img src="https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/images/tmptvkkr0tg.png" width="20%" height="20%"> <img src="https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/images/inpaint_seg.jpg" width="20%" height="20%">

You can also use pose as control hint. But please note that it is suggested to use OpenPose format, which is consistent to the training process. If you just want to test a few images without install OpenPose locally, you can directly use [online demo of ControlNet](https://huggingface.co/spaces/hysts/ControlNet) to generate pose image given the resized 512x512 input.

```
image = load_image("./images/pose_image.jpg")
mask = load_image("./images/pose_mask.jpg")
pose_image = load_image('./images/pose_hint.png')

image = pipe_control(prompt="Face of a young boy smiling", 
                     negative_prompt="lowres, bad anatomy, worst quality, low quality",
                     controlnet_hint=pose_image, 
                     image=image,
                     mask_image=mask,
                     num_inference_steps=100).images[0]

image.save("inpaint_pos.jpg")
```

<img src="https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/images/pose_image.jpg" width="20%" height="20%"> <img src="https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/images/pose_mask.jpg" width="20%" height="20%"> <img src="https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/images/pose_hint.png" width="20%" height="20%"> <img src="https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/images/inpaint_pos.jpg" width="20%" height="20%">

# ControlNet + Inpainting + Img2Img
We have uploaded [pipeline_stable_diffusion_controlnet_inpaint_img2img.py](https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/pipeline_stable_diffusion_controlnet_inpaint_img2img.py) to support img2img. You can follow the same instruction as [this section](https://github.com/haofanwang/ControlNet-for-Diffusers#controlnet--inpainting).

# Multi-ControlNet (experimental)

Add two controlNets to the multicontrolnet pipeline.

```
cp pipeline_stable_diffusion_multi_controlnet_inpaint.py  PATH/pipelines/stable_diffusion
```
First I copied the unet from inpainting model and replaced the unet of control_sd15_depth model with it and called the new folder control_sd15_depth_inpaint.

Then I updated the current file "pipeline_stable_diffusion_controlnet_inpaint.py" to take in two control inputs and their weights.

After that I added controlnet2 to the pipe_control and set weights for the controls. It is now working.

```
controlnet2_path= "models/control_sd15_scribble"  # 
controlnet2 = UNet2DConditionModel.from_pretrained(controlnet2_path, subfolder="controlnet").to("cuda")
pipe_control = StableDiffusionControlNetInpaintPipeline.from_pretrained("models/control_sd15_depth_inpaint",controlnet2=controlnet2,torch_dtype=torch.float16).to('cuda')
pipe_control.unet.in_channels = 4
pipe_control.enable_attention_slicing()

output_image  = pipe_control(prompt=prompt, 
                                negative_prompt="human, hands, fingers, legs, body parts",
                                image=image,
                                mask_image=mask,
                                controlnet_hint1=control_image_1, 
                                controlnet_hint2=control_image_2, 
                                control1_weight=1, # Default is 1, you can change this if need be
                                control2_weight=1, # Default is 1, you can change this if need be
                                height=height,
                                width=width,
                                generator=generator,
                                num_inference_steps=100).images[0]


```
If you want to add more than 2 control nets into the pipeline, 
Open the pipeline file and replace:

```
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: UNet2DConditionModel,
        controlnet2:UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            controlnet2=controlnet2,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
```

With

```

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: UNet2DConditionModel,
        controlnet2:UNet2DConditionModel,
        controlnet3:UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            controlnet2=controlnet2,
            controlnet3=controlnet3,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
                
```

Replace:

```
def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_hint1: Optional[Union[torch.FloatTensor, np.ndarray, PIL.Image.Image]] = None,
        controlnet_hint2: Optional[Union[torch.FloatTensor, np.ndarray, PIL.Image.Image]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        control1_weight: Optional[float] = 1.0,
        control2_weight: Optional[float] = 1.0,
    ):
    ```
    
    with
    ```
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_hint1: Optional[Union[torch.FloatTensor, np.ndarray, PIL.Image.Image]] = None,
        controlnet_hint2: Optional[Union[torch.FloatTensor, np.ndarray, PIL.Image.Image]] = None,
        controlnet_hint3: Optional[Union[torch.FloatTensor, np.ndarray, PIL.Image.Image]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        control1_weight: Optional[float] = 1.0,
        control2_weight: Optional[float] = 1.0,
        control3_weight: Optional[float] = 1.0,
    ):
    ```
    
Add:

```
        # 1. Control Embedding check & conversion
        
        ...
        
        if controlnet_hint3 is not None:
            controlnet_hint3 = self.controlnet_hint_conversion(controlnet_hint3, height, width, num_images_per_prompt)

And replace:

 ```
 if controlnet_hint1 is not None:
                    # ControlNet predict the noise residual
                    merged_control = []

                    control1 = self.controlnet(
                        latent_model_input, t, encoder_hidden_states=prompt_embeds, controlnet_hint=controlnet_hint1
                    )

                    if controlnet_hint2 is not None:    
                        control2 = self.controlnet(
                            latent_model_input, t, encoder_hidden_states=prompt_embeds, controlnet_hint=controlnet_hint2
                        )
                        


                            for i in range(len(control1)):
                                merged_control.append(control1_weight*control1[i]+control2_weight*control2[i])                    

                            control = merged_control

                    else:

                        control = control1
```

 with 

```
      if controlnet_hint1 is not None:
                    # ControlNet predict the noise residual
                    merged_control = []

                    control1 = self.controlnet(
                        latent_model_input, t, encoder_hidden_states=prompt_embeds, controlnet_hint=controlnet_hint1
                    )

                    if controlnet_hint2 is not None:    
                        control2 = self.controlnet2(
                            latent_model_input, t, encoder_hidden_states=prompt_embeds, controlnet_hint=controlnet_hint2
                        )
                        
                       if controlnet_hint3 is not None:    
                         control3 = self.controlnet3(
                             latent_model_input, t, encoder_hidden_states=prompt_embeds, controlnet_hint=controlnet_hint3
                         )

                         for i in range(len(control1)):
                             merged_control.append(control1_weight*control1[i]+control2_weight*control2[i]+control3_weight*control3[i])                    

                          control = merged_control
                          
                        else:
                        
                        
                         for i in range(len(control1)):
                             merged_control.append(control1_weight*control1[i]+control2_weight*control2[i])                    

                          control = merged_control
                    
                    else:

                        control = control1
                        
 ```
 

 



# Acknowledgement
We first thanks the author of [ControlNet](https://github.com/lllyasviel/ControlNet) for such a great work, our converting code is borrowed from [here](https://github.com/lllyasviel/ControlNet/discussions/12). We are also appreciated the contributions from this [pull request](https://github.com/huggingface/diffusers/pull/2407) in diffusers, so that we can load ControlNet into diffusers.

# Contact
The repo is still under active development, if you have any issue when using it, feel free to open an issue.
