import json
import torch
from tqdm import trange
from transformers import AutoModel, AutoTokenizer
from omegaconf import OmegaConf
from cd_metric import cd_metric
from sudoku_metric import is_valid_sudoku
from trip_metric import trip_metric

EVAL_DIR = "data"

class Generator():
    def __init__(self, model, tokenizer, gen_kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_kwargs = gen_kwargs
    
    def generate(self, inputs):
        batch_size = 1
        responses = []
        for i in trange(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            model_inputs = self.tokenizer(
                batch, 
                padding=True, 
                padding_side="left",
                truncation=False, 
                return_tensors="pt"
            ).to(self.model.device)
            # print(model_inputs)
            with torch.no_grad():
                generated_ids = self.model.diffusion_generate(
                    inputs=model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    **self.gen_kwargs
                )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            responses.extend(response)
        return responses


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def write_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

def eval_cd3(generator, prediction_path=None):
    data = read_jsonl(EVAL_DIR+"/cd3_test.jsonl")
    n_few_shots = 8
    template = "Given 4 numbers, use +-*/ to operate over the first three numbers to achieve the last number.\n\n"
    template += "\n\n".join([f"Input: {i['input']}\nOutput: {i['output']}" for i in data[:n_few_shots]]) \
        + "\n\nInput: {input}\nOutput: "
    data = data[n_few_shots:]

    inputs = [template.format(input=i['input']) for i in data]
    print("Example input: ", inputs[0])
    generations = generator.generate(inputs)
    generations = [g.split('<|end_of_text|>')[0].split('\n')[0] for g in generations]
    print("Acc: ", cd_metric([i['input'] for i in data], generations))
    if prediction_path is not None:
        write_jsonl([{"input": i['input'], "gold": i['output'], "prediction": j} for i, j in zip(data, generations)], prediction_path)


def eval_cd4(generator, prediction_path=None):
    data = read_jsonl(EVAL_DIR+"/cd4_test.jsonl")
    n_few_shots = 8
    template = "Given 5 numbers, use +-*/ to operate over the first four numbers to achieve the fifth number.\n\n"
    template += "\n\n".join([f"Input: {i['input']}\nOutput: {i['output']}" for i in data[:n_few_shots]]) \
        + "\n\nInput: {input}\nOutput: "
    data = data[n_few_shots:]

    inputs = [template.format(input=i['input']) for i in data]
    print("Example input: ", inputs[0])
    generations = generator.generate(inputs)
    generations = [g.split('<|end_of_text|>')[0].split('\n')[0] for g in generations]
    print("Acc: ", cd_metric([i['input'] for i in data], generations))
    if prediction_path is not None:
        write_jsonl([{"input": i['input'], "gold": i['output'], "prediction": j} for i, j in zip(data, generations)], prediction_path)


def eval_cd5(generator, prediction_path=None):
    data = read_jsonl(EVAL_DIR+"/cd5_test.jsonl")
    n_few_shots = 8
    template = "Given 6 numbers, use +-*/ to operate over the first five numbers to achieve the last number.\n\n"
    template += "\n\n".join([f"Input: {i['input']}\nOutput: {i['output']}" for i in data[:n_few_shots]]) \
        + "\n\nInput: {input}\nOutput: "
    data = data[n_few_shots:]

    inputs = [template.format(input=i['input']) for i in data]
    print("Example input: ", inputs[0])
    generations = generator.generate(inputs)
    generations = [g.split('<|end_of_text|>')[0].split('\n')[0] for g in generations]
    print("Acc: ", cd_metric([i['input'] for i in data], generations))
    if prediction_path is not None:
        write_jsonl([{"input": i['input'], "gold": i['output'], "prediction": j} for i, j in zip(data, generations)], prediction_path)


def eval_sudoku(generator, prediction_path=None):
    # for n in [4,5,6,7,8,9,10,11,12]:
    for n in [10]:
        data = read_jsonl(EVAL_DIR+f"/sudoku_4x4_{n}.jsonl")
        n_few_shots = 8
        template = "Fill the positions where the values are 0 in a 4x4 grid with digits 1-4 so that each column, each row, and each of the four 2x2 subgrids that compose the grid contains all of the digits from 1 to 4.\n\n"
        template += "\n\n".join([f"Input:\n{i['input']}\nOutput:\n{i['output']}" for i in data[:n_few_shots]]) \
            + "\n\nInput:\n{input}\nOutput:\n "
        data = data[n_few_shots:]

        inputs = [template.format(input=i['input']) for i in data]
        print("Example input: ", inputs[0])
        generations = generator.generate(inputs)
        generations = [g.split('<|endoftext|>')[0].split('\n\n')[0].replace(' ', '') for g in generations]
        # generations = [i["prediction"] for i in read_jsonl(prediction_path+f"_{n}")]
        acc = sum([is_valid_sudoku(i['input'], j) for i, j in zip(data, generations)]) / len(data)
        print(f"Filled n={n}, Acc: ", acc)
        if prediction_path is not None:
            write_jsonl([{"input": i['input'], "gold": i['output'], "prediction": j} for i, j in zip(data, generations)], prediction_path+f"_{n}")



def eval_trip(generator, prediction_path=None):
    with open(EVAL_DIR+"/trip_planning.json") as f:
        data = json.load(f)
        
    # two shots
    inputs = []
    for i in data.values():
        splits = i['prompt_5shot'].split("TASK:")
        inputs.append("TASK:".join(splits[:3]+[splits[-1]]))

    print("Example input: ", inputs[0])
    generations = generator.generate(inputs)
    generations = [g.split('<|endoftext|>')[0].split('\n\nTASK')[0] for g in generations]
    # generations = [i["prediction"] for i in read_jsonl(prediction_path)]
    if prediction_path is not None:
        write_jsonl([{"input": i["prompt_0shot"], "gold": i['golden_plan'], "prediction": j} for i, j in zip(data.values(), generations)], prediction_path)
    trip_metric(data, generations)


if __name__ == "__main__":
    # Load CLI arguments (overrides) and combine with a YAML config
    cfg = OmegaConf.from_cli()
    default = OmegaConf.create({
        'task': 'sudoku',
        'prediction_path': 'sudoku_result',
        'ckpt': 'Dream-org/Dream-v0-Base-7B',
        'gen_params': {
                'diffusion_steps': 24, 
                'max_new_tokens': 24,
                'temperature': 0,
                'top_p': 1
            }
    })
    cfg = OmegaConf.merge(default, cfg)
    model = AutoModel.from_pretrained(cfg.ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.ckpt, trust_remote_code=True)
    model = model.cuda()
    generator = Generator(model, tokenizer, cfg.gen_params)
    print(cfg)
    eval(f"eval_{cfg.task}")(generator, cfg.prediction_path)
