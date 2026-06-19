# PolyMentor Hugging Face Deployment Guide

This guide covers two deployments:

- Hugging Face Hub: publish the optional local LoRA checkpoint.
- Hugging Face Spaces: run a public chatbot demo.

Groq remains the easiest production runtime. The local LoRA checkpoint is for
experiments and demonstrations.

## 1. Prepare Accounts And Tools

Create or log into a Hugging Face account, then install the CLI:

```bash
python -m pip install -U huggingface_hub
hf auth login
```

For Spaces, you can deploy through Git. Make sure Git LFS is installed if you
push large model files.

## 2. Fine-Tune Locally

Use Python 3.12 for CUDA PyTorch on Windows:

```powershell
deactivate
py -3.12 -m venv venv312
.\venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
bash scripts/train.sh
```

Expected output:

```text
models_saved/polymentor-chatbot-lora
```

## 3. Publish Checkpoint To HF Hub

Create a model repo:

```bash
hf repo create polymentor-chatbot-lora --type model
```

Upload the adapter folder:

```bash
hf upload your-username/polymentor-chatbot-lora models_saved/polymentor-chatbot-lora . --repo-type model
```

Recommended model card sections:

- Base model, for example `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- Training data summary
- Intended use: coding tutor, bug explanation, practice help
- Limitations: not guaranteed correct, verify generated code
- Safety: do not paste secrets, credentials, or private source code

## 4. Create A Hugging Face Space

Go to Hugging Face -> New Space.

Good first choice:

- SDK: Gradio
- Hardware: CPU Basic for Groq API usage
- Visibility: public or private

If you later run a local model inside the Space, choose GPU hardware.

## 5. Add Space Secrets

In Space settings, add:

```text
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.3-70b-versatile
```

Never commit `.env` or API keys to the Space repo.

## 6. Minimal Space Files

`requirements.txt`:

```text
gradio
groq>=1.4.0
pydantic>=2.7
```

`app.py`:

```python
import os
import gradio as gr
from groq import Groq

MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM = (
    "You are PolyMentor, a coding tutor chatbot. Teach programming, identify "
    "likely bugs, explain fixes, and provide corrected code when useful."
)


def mentor(question, code, language, level):
    user = (
        f"Learner level: {level}\nLanguage: {language}\n"
        f"Question: {question}\n\nCode:\n```{language}\n{code}\n```"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.25,
        max_completion_tokens=1800,
    )
    return response.choices[0].message.content


demo = gr.Interface(
    fn=mentor,
    inputs=[
        gr.Textbox(label="Question", value="Find the bug and teach me the concept"),
        gr.Code(label="Code"),
        gr.Dropdown(["python", "javascript", "typescript", "java", "cpp", "go", "rust"], value="python", label="Language"),
        gr.Dropdown(["beginner", "intermediate", "advanced"], value="beginner", label="Level"),
    ],
    outputs=gr.Markdown(label="PolyMentor"),
    title="PolyMentor",
    description="Groq-powered coding tutor chatbot.",
)

if __name__ == "__main__":
    demo.launch()
```

## 7. Push The Space

```bash
git clone https://huggingface.co/spaces/your-username/polymentor-space
cd polymentor-space
# add app.py and requirements.txt
git add .
git commit -m "Deploy PolyMentor Space"
git push
```

The Space will build automatically after push.

## 8. Deployment Checklist

- `GROQ_API_KEY` is stored as a Space secret.
- The app does not print secrets.
- The demo says outputs should be verified.
- The model card explains limitations.
- The Space README tells users not to paste private code or credentials.

## Official References

- Hugging Face Hub upload docs: https://huggingface.co/docs/huggingface_hub/guides/upload
- Hugging Face Spaces overview: https://huggingface.co/docs/hub/spaces
- Spaces configuration: https://huggingface.co/docs/hub/spaces-config-reference
