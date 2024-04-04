# deception-detection 🕵🏽‍♂️
## Detecting lying signals using LLM and RAG
This repository conatins codes and descriptions of the deception detection project conducted as a part of BITAmin🍊 conference. The goal of this project is to detect signs of deception in conversations between an investigator and a suspect. To achieve this, we first utilized GPT-3.5 to generate investigator-suspect dialogue scripts containing lying signals in the suspect's utterances. Subsequently, we trained the LLaMA-2 model using this data as training data. Additionally, we constructed a RAG model which accepts incident records as external knowledge to identify inconsistencies between the conversation content and the incident records. Please refer to the `presentation.pdf` file in the `others` folder for detailed information about the project. Below are the frameworks used for the project. 

<div align="left">
 <img src = "https://img.shields.io/badge/LangChain-blue?style=flat-square"/>
 <img src = "https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=SciPy&logoColor=white"/>
 <img src = "https://img.shields.io/badge/FFmpeg-007808?style=flat-square&logo=FFmpeg&logoColor=white"/>
 <img src = "https://img.shields.io/badge/GoogleColab-F9AB00?style=flat-square&logo=Google%20Colab&logoColor=white"/>
 <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=NumPy&logoColor=white"/>
 <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>
</div>

 ## 1. Code Description 📝
### synthetic-data-generation.ipynb
This script is designed for generating synthetic conversation data, specifically focusing on dialogues between an investigator and a suspect. Note that I refered to [official guidance of LangChain](https://python.langchain.com/docs/use_cases/data_generation) for this stage. Below is an explanation of the different sections and functionalities within the script:
  1. **Loading Dataset and Preparing Prompt**: The script loads a dataset from a CSV file named `contradicts.csv`. This file lists two contradictory sentences and was used to include contradictory sentences in the suspect's speech. (We obtained the data from [here](https://www.kaggle.com/datasets/athu1105/contradiction-detection?select=train.csv)) Next, the code defines classes and configurations necessary for generating synthetic data. It sets up a template for generating conversation between an investigator and a suspect. This includes defining the conversation structure, the types of lying signals (`IH_A`, `IH_B`, `VE`, `LM`, `TP`), and example prompts illustrating how each type of lying signals is used.
  2. **Synthetic Data Generation**: This section utilizes an OpenAI model (GPT-3.5-turbo) through the LangChain library to generate synthetic conversation data based on the constructed templates.
  3. **Preprocessing**: Remove the lying siganl tags from the conversation script to build train dataset. 

-----

### fine-tuning.ipynb
This script uses `autotrain-advanced` package for fine-tuning LLaMA-2 model. Note that I refered to [official guidance of hugging face](https://github.com/huggingface/autotrain-advanced?tab=readme-ov-file) for this stage. Below is an detailed explanation of the different sections from the code. 
   1. **Setting Hyperparameters**: Various hyperparameters such as model name, learning rate, number of epochs, batch size, etc., are configured. These settings play a crucial role in the model performance.
   2. **Fine-tuning**: Using autotrain, the script fine-tunes LLaMA-2 model with the provided hyperparameters. In this step, the `autotrain` command is used to:
- Set up the model and data paths
- Pass hyperparameters like learning rate, batch size, number of epochs, etc.
- Activate features such as token push, quantization, mixed precision, etc., based on options.

-----

### deception-detection.ipynb
This script utilized `LangChain` library and fine-tuned LLaMA model to construct RAG. Below is an detailed explanation of the different sections from the code.
   1. **Setting Retrievals**: Loads a PDF document (`investigation-report.pdf`) and splits it into pages. Utilizes a `RecursiveCharacterTextSplitter` to chunk the pages into smaller text chunks. Embeds the text chunks using `OpenAIEmbeddings` and stores them in a vector store using `Chroma`.
   2. **Model Inference with RAG**: Sets up a retriever to search through the stored text chunks. Given an instruction and input conversation script, the script queries the model to identify lying signals in the suspect's utterances.
-----

## 2. Awards 🏆
🥇 Grand Prize
