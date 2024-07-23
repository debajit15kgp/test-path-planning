from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline
import os

os.environ['TORCH_USE_CUDA_DSA'] = "1"
model_id = "google/ddpm-cifar10-32"

# load model and scheduler
# ddpm = DDPMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference

image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
image_pipe.to("cuda")

# run pipeline in inference (sample random noise and denoise)
images = image_pipe().images
# image = ddpm().images[0]

# save image
images[0].save("ddpm_generated_image.png")