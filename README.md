# deception-detection 🕵🏽‍♂️
## Detecting lying signals using LLM and RAG models
This repository conatins codes and descriptions of the deception detection project conducted as a part of 🍊BITAmin conference. Below are the frameworks used for the project. 

<div align="left">
 <img src = "https://img.shields.io/badge/langchain-blue?style=flat-square"/>
 <img src = "https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=SciPy&logoColor=white"/>
 <img src = "https://img.shields.io/badge/FFmpeg-007808?style=flat-square&logo=FFmpeg&logoColor=white"/>
 <img src = "https://img.shields.io/badge/GoogleColab-F9AB00?style=flat-square&logo=Google%20Colab&logoColor=white"/>
 <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=NumPy&logoColor=white"/>
 <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>
</div>

 ## 1. Code Description 📝
### synthetic-data-generation.ipynb
This script is designed for generating synthetic conversation data, specifically focusing on dialogues between an investigator and a suspect. Note that I refered to [official guidance of LangChain](https://python.langchain.com/docs/use_cases/data_generation) for this stage. Below is an explanation of the different sections and functionalities within the script:
  1. **Loading Dataset and Preparing Prompt**: The script loads a dataset from a CSV file named `contradicts.csv`. It also defines classes and configurations necessary for generating synthetic data. It sets up a template for generating conversation prompts between an investigator and a suspect. This includes defining the conversation structure, the types of utterances (`IH_A`, `IH_B`, `VE`, `LM`, `TP`), and example prompts illustrating how each type of utterance is used.
  2. **Synthetic Data Generation**: This script utilizes an OpenAI model (GPT-3.5-turbo) through the LangChain library to generate synthetic conversation data based on the constructed templates.

-----

### fine-tuning.ipynb
This script uses `autotrain-advanced` package for fine-tuning LLaMA-2 model. Note that I refered to [official guidance of hugging face](https://github.com/huggingface/autotrain-advanced?tab=readme-ov-file) for this stage. Below is an detailed explanation of the different sections from the code. 
   1. **Setting Hyperparameters**: Various hyperparameters such as model name, learning rate, number of epochs, batch size, etc., are configured. These settings play a crucial role in the performance and tuning of the training process.
   2. **Fine-tuning**: Using Autotrain, the script fine-tunes the specified model with the provided hyperparameters. In this step, the `autotrain` command is used to:
 - Set up the model and data paths
 - Pass hyperparameters like learning rate, batch size, number of epochs, etc.
 - Activate features such as token push, quantization, mixed precision, etc., based on options.

-----

### deception-detection.ipynb
This script utilized `LangChain` library and fine-tuned LLaMA model to construct RAG program. Below is an detailed explanation of the different sections from the code.
   1. **Setting Retrievals**: Loads a PDF document (`investigation-report.pdf`) and splits it into pages. Utilizes a `RecursiveCharacterTextSplitter` to chunk the pages into smaller text chunks. Embeds the text chunks using `OpenAIEmbeddings` and stores them in a vector store using `Chroma`.
   2. **Model Inference with RAG**: Sets up a retriever to search through the stored text chunks. Given an instruction and input conversation script, the script queries the model to identify lying signals in the suspect's utterances.
-----

## 2. Awards 🏆
🥇 Grand Prize
