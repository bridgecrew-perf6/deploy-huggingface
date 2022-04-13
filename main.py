import torch, transformers, gdown, gc
from flask import Flask, request, make_response, jsonify

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# download model
id = "1zLsKoVDIAZUrqm2DFendSVNMguL7fA9L"
gdown.download_folder(id=id, quiet=False, use_cookies=False)
modelpath = "."
# load tokenizer and model from disk
tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
model = transformers.AutoModelForCausalLM.from_pretrained(modelpath, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

# Flask app
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# generate messages from huggingface
def get_message(inputSTR,num_return_sequences=1,max_new_tokens=50,temperature=0.9):
  inputs = tokenizer(inputSTR, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
  outputs = model.generate(inputs, no_repeat_ngram_size=5, max_new_tokens=max_new_tokens, top_p=1 , temp=temperature , do_sample=True , num_return_sequences=num_return_sequences)
  message = [{'text':tokenizer.decode(output)} for output in outputs]
  return message

# set CORS policy to allow all origins
def _build_cors_prelight_response():
    response = make_response()
    [response.headers.add(tag, "*") for tag in ["Access-Control-Allow-Origin", "Access-Control-Allow-Headers", "Access-Control-Allow-Methods"]]
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

# API endpoint to generate messages
@app.route('/complete', methods=['POST', 'OPTIONS'])
def complete():
    gc.collect()
    if request.method == "OPTIONS":  # CORS preflight
        return _build_cors_prelight_response()
    elif request.method == "POST":  # The actual request following the preflight
        content = request.json
        response = get_message(content["prompt"],content["n"],content["max_tokens"],content["temperature"])
        return _corsify_actual_response(jsonify({"choices": response}))
    else:
        raise RuntimeError("Weird - don't know how to handle method {}".format(request.method))

app.run()