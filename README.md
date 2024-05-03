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

## Usage

Render images:

```bash
blender <path_to_blend_file> --background --python blender_script.py
```

Generate questions and answers:

```bash
python generate_qa.py
```

## Contributing

Ask questions by creating an issue.

PRs accepted.

## License

MIT © Andy Zijian Zhang
