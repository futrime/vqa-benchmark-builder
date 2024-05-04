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

Render images:

```bash
blender <path_to_blend_file> --background --python blender_script.py
```

Generate the other parts of the dataset:

```bash
python generate_dataset.py
```

Finetune the generator:

```bash
source finetune_generator.sh
```

## Contributing

Ask questions by creating an issue.

PRs accepted.

## License

MIT © Andy Zijian Zhang
