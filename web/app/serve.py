import argparse
import json
import logging
import time

import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException

from fastchat_prompt import FastChatLLM

logging.basicConfig(level=logging.INFO)

app = FastAPI()
router = APIRouter()

'''
{
"messages": [
 {"role": "user", "content": "What is 300 times 1000"},
 {"role": "assistant", "content": "300 multiplied by 1000 is 300000."},
 {"role": "user", "content": "Who are you?"},
],
"model": "dhabi-7b",
}
'''


@router.get("/")
async def home():
    return {"message": "MBZUAI Chat bot service"}

app.include_router(router)


def parse_model_args(model_args_list):
    try:
        return sorted(model_args_list, key=lambda sort_key: int(sort_key["order"]))
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            f"Invalid json in '--register-model' arguments {model_args_list}. 'order' property is required and must be a valid int.") from error


'''
python serve.py '--host' 0.0.0.0 '--port' '11111' '--register-model' '{"path": "/models/13B", "name": "dhabi-13b", "order": 3}' '--register-model' '{"path": "/models/dhabi_identity_7B", "name": "dhabi-7b", "order": 4}' '--register-model' '{"path": "/models/dhabi_llama2_13B", "order": 1}' '--register-model' '{"path": "/models/dhabi_llama2_7B", "order": "2"}'
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--register-model", type=json.loads, action="append", required=True)
    args = parser.parse_args()

    chatbot_name_map = {}
    for model in parse_model_args(args.register_model):
        # Validate json schema
        if "path" not in model or not model["path"]:
            raise RuntimeError(
                f"Invalid json in '--register-model' argument {model}. 'path' property is required and cannot be empty.")
        if "name" in model and not model["name"]:
            raise RuntimeError(f"Invalid json in '--register-model' argument {model}. 'name' property cannot be empty.")
        # Populate chatbot_name_map
        model_path = model["path"][:-1] if model["path"].endswith("/") else model["path"]
        model_name = model.get("name", model_path.split("/")[-1])
        model_device = model.get("device", "cuda")
        model_num_gpus = model.get("num_gpus", "4")
        model_max_memory = model.get("max_memory", {})
        model_conv_template = model.get("conv_template", "vicuna_v1.1")
        model_temperature = model.get("temperature", 0.7)
        model_max_new_tokens = model.get("max_new_tokens", 512)
        model_debug = model.get("debug", False)
        chatbot_name_map[model_name] = FastChatLLM(model_path=model_path, device=model_device, num_gpus=model_num_gpus
                                                   , max_memory=model_max_memory, conv_template=model_conv_template, temperature=model_temperature
                                                   , max_new_tokens=model_max_new_tokens, debug=model_debug)

    if not chatbot_name_map:
        raise RuntimeError(f"Register a model using the '--register-model' argument.")
    uvicorn.run(app, host=args.host, port=args.port)
