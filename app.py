import torch
from diffusers import StableDiffusionPipeline
import gradio as gr
import time
import sys

# Setting up the AI model and telling it to use your M1 chip's GPU
model_id = "runwayml/stable-diffusion-v1-5"
device = "mps" # This taps into your M1's graphics power

print("Initializing model in full precision... This may take a moment on first run.")
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)
pipe.enable_attention_slicing()
print("Model loaded and memory optimization enabled.")

# The main function that creates your image and shows you how it's progressing

def generate_image_with_progress(prompt, width, height, num_inference_steps, guidance_scale, seed, progress=gr.Progress()):
    """
    This is where the magic happens - your prompt becomes an actual image!
    """
    start_time = time.time()
    progress(0, desc="Starting...")

    generator = torch.Generator(device).manual_seed(int(seed))

    def progress_callback(pipe, step, timestep, callback_kwargs):
        current_progress = (step + 1) / num_inference_steps
        elapsed_time = time.time() - start_time
        
        if current_progress > 0:
            estimated_total_time = elapsed_time / current_progress
            eta_seconds = estimated_total_time - elapsed_time
            eta_message = f"ETA: {eta_seconds:.1f}s" if eta_seconds > 0 else "Finishing..."
        else:
            eta_message = "Calculating ETA..."

        progress(current_progress, desc=f"Step {step+1}/{num_inference_steps} ({eta_message})")
        return callback_kwargs

    print(f"Generating image for prompt: '{prompt}'")
    
    image = pipe(
        prompt=prompt,
        width=int(width),
        height=int(height),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=guidance_scale,
        generator=generator,
        callback_on_step_end=progress_callback,
    ).images[0]

    end_time = time.time()
    print(f"Image generation complete in {end_time - start_time:.2f} seconds.")
    return image

# Building the web interface where you'll interact with everything

with gr.Blocks() as demo:
    gr.Markdown("# M1 AI Image Generator 🎨")
    gr.Markdown("Generate images using Stable Diffusion. Dimensions must be divisible by 8.")
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="Image Prompt",
                value="A majestic fantasy castle on a mountain top, vibrant sunset, digital painting",
                lines=3
            )
            with gr.Row():
                steps_slider = gr.Slider(minimum=10, maximum=100, value=50, step=1, label="Inference Steps")
                guidance_slider = gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.5, label="Guidance Scale")
            with gr.Row():
                # Using 512x512 by default - it's the sweet spot for speed and quality
                width_slider = gr.Slider(minimum=256, maximum=1024, value=512, step=8, label="Width")
                height_slider = gr.Slider(minimum=256, maximum=1024, value=512, step=8, label="Height")
            
            seed_number = gr.Number(value=42, label="Seed (change for a new image)")
            generate_button = gr.Button("Generate Image", variant="primary")
            
        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image", type="pil")

    gr.Examples(
        examples=[
            ["A cozy cabin in a snowy forest, northern lights, soft lighting, detailed", 512, 512, 50, 7.5, 123],
            ["An astronaut riding a unicorn on the moon, cinematic, 4k", 512, 512, 50, 7.5, 456],
        ],
        inputs=[prompt_input, width_slider, height_slider, steps_slider, guidance_slider, seed_number]
    )
    
    generate_button.click(
        fn=generate_image_with_progress,
        inputs=[prompt_input, width_slider, height_slider, steps_slider, guidance_slider, seed_number],
        outputs=output_image
    )

# Time to fire up the app and start creating!
print("\nStarting Gradio app...")
demo.launch()