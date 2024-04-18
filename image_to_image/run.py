import os
import base64
import logging
import requests
from glob import glob
from PIL import Image
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from image_to_image.schemas import InputSchema


load_dotenv()
STABILITY_API_HOST = "https://api.stability.ai"
DEFAULT_FILENAME = "output.png"
DEFAULT_ENGINE = "stable-diffusion-xl-1024-v1-0"

def get_logger(__name__):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = get_logger(__name__)

def run(job: InputSchema, cfg: dict = None, **kwargs):
    logger.info(f"Running job with prompt: {job.prompt}")

    # Get api key from environment variable
    api_key = os.environ['STABILITY_KEY']

    if api_key is None:
        raise ValueError("API key is not set")

    # Set the data
    url = f"{STABILITY_API_HOST}/v1/generation/{DEFAULT_ENGINE}/image-to-image"

    
    if job.input_dir:
        # read the folder and get the first image
        image_path = glob(f"{job.input_dir}/*")[0]

        # open and resize the image 1024x1024
        image = Image.open(image_path)
        image = image.resize((1024, 1024))

        # save to tempfile
        image_path = "/tmp/init_image.png"
        image.save(image_path)

        files={
            "init_image": open(image_path, "rb")
        }

    elif job.image:
        # convert from base64 to image
        image = Image.open(BytesIO(base64.b64decode(job.image)))

        # resize the image 1024x1024
        image = image.resize((1024, 1024))
        
        # add to tempfile 
        image_path = "/tmp/init_image.png"
        image.save(image_path)

        files={
            "init_image": open(image_path, "rb")
        }
    else:
        raise ValueError("No image provided")
    

    
    response = requests.post(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        files=files,
        data={
            "image_strength": 0.35,
            "init_image_mode": "IMAGE_STRENGTH",
            "text_prompts[0][text]": job.prompt,
            "cfg_scale": 7,
            "samples": 1,
            "steps": 30,
        }
    )

    if response.status_code != 200:
        logger.error(f"Failed to generate image: {response.text}")
        raise ValueError(f"Failed to generate image: {response.text}")
    
    result = response.json()

    image_b64 = result['artifacts'][0]['base64']
    image = Image.open(BytesIO(base64.b64decode(image_b64)))

    if job.output_path:
        output_path = job.output_path
        Path(output_path).mkdir(parents=True, exist_ok=True)
        image.save(f"{output_path}/{DEFAULT_FILENAME}")

        return f"Image saved to {output_path}/{DEFAULT_FILENAME}"

    return "Image generated successfully"


if __name__ == "__main__":
    # try with image_path
    input = InputSchema(
        prompt="A beautiful sunset over the ocean",
        input_dir="./input_folder",
        output_path="output_folder"
    )

    run(input)

    # try with image
    base64_image = base64.b64encode(open("./input_folder/image.jpeg", "rb").read()).decode("utf-8")
    input = InputSchema(
        prompt="A beautiful sunrise over the ocean",
        image=base64_image,
        output_path="output_folder"
    )

    run(input)