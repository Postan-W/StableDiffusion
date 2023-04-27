from torch import autocast
import torch
from diffusers import StableDiffusionPipeline
from utils import get_next
"""
本地运行显存不足:
If you are limited by GPU memory and have less than 4GB of GPU RAM available, 
please make sure to load the StableDiffusionPipeline in float16 precision instead of the 
default float32 precision as done above. You can do so by telling diffusers to expect the weights to be 
in float16 precision.
一些试验表明float16并未降低准确度。
"""
pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path="../model/stable-diffusion-v1-4",torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()#attention顺序运行。没有该句则是批处理，占显存较大
generator = torch.Generator("cuda").manual_seed(2147483647)#固定seed，这样同样的prompt生成的结果接近。(如果初始分布数据一致的话那么结果完全相同)
prompt = "virago sleeve dress, lotus root shaped sleeves, intricate fabric details, fashion product catalog image, studio lighting, front view, square image"
with autocast("cuda"):
    # image = pipe(prompt,generator=generator).images[0]
    # image.save("./images/" + get_next("./images",".jpg"))
    print(len(pipe(prompt,generator=generator).images))