from torch import autocast
import torch
from diffusers import StableDiffusionPipeline,DPMSolverMultistepScheduler
from utils import get_next

pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path="../model/stable-diffusion-v1-4",torch_dtype=torch.float16)#使用16bits占用显存更小、计算更快，效果差别不大
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()#attention顺序运行。没有该句则是批处理，占显存较大
#scheduler参见https://huggingface.co/docs/diffusers/main/en/api/schedulers/overview
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
generator = torch.Generator("cuda").manual_seed(2147483647)#固定seed，这样同样的prompt生成的结果接近。(如果初始分布数据一致的话那么结果完全相同)


with autocast("cuda"):
    prompt = "a white t-shirt with the slogan I love stable diffusion"
    image = pipe(prompt,generator=generator,num_inference_steps=30).images[0]
    image.save("./images/" + get_next("./images",".jpg"))
