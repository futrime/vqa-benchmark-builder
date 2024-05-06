# vqa

A VQA project

## Install

We tested the code on the following environment:

- Ubuntu 22.04.4 LTS
- Python 3.10.14
- Blender 2.93.18

See other package versions in `requirements.txt`.

To install the required packages, run:

```bash
pip install -r requirements.txt
```

Install packages from submodule LLaVA:

```bash
pip install --upgrade pip
pip install -e third_party/LLaVA[train]
pip install flash-attn --no-build-isolation
```

## Usage

Generate dataset:

```bash
blender <path_to_blend_file> --background --python generate_dataset_images.py
python generate_dataset.py
```

Finetune the generator:

```bash
python convert_dataset.py
source finetune_generator.sh
python ./third_party/LLaVA/scripts/merge_lora_weights.py --model-path ./data/checkpoints/llava-v1.5-7b-task-lora-generator/ --model-base liuhaotian/llava-v1.5-7b --save-model-path ./data/models/llava-v1.5-7b-task-lora-generator
```

Evaluate the generator and generate results:

```bash
python evaluate_generator.py
```

Finetune the verifier:
    
```bash
python convert_generation_results.py
source finetune_verifier.sh
python ./third_party/LLaVA/scripts/merge_lora_weights.py --model-path ./data/checkpoints/llava-v1.5-7b-task-lora-verifier/ --model-base ./data/models/llava-v1.5-7b-task-lora-generator --save-model-path ./data/models/llava-v1.5-7b-task-lora-verifier
```

## Contributing

Ask questions by creating an issue.

PRs accepted.

## License

MIT © Andy Zijian Zhang
