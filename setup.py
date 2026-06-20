from setuptools import setup, find_packages

setup(
    name="polymentor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "groq",
        "pydantic",
        "pymongo",
        "python-dotenv",
        "uvicorn",
    ],
)
