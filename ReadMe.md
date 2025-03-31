# **Introduction**

Quanta Chatbot is a Python application designed to handle multiple files, including PDFs, for in-depth discussions. It specializes in working with research papers or lengthy documents, providing precise and contextually relevant response to user questions based on document content(s).

## **Installation**

Follow the steps below to set up your chatbot:

1. **Install Python**

   If you have never used Python on your computer before, make sure to download one.

2. **Install dependencies**

   Run the following command to install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install other files**

   Sometimes, manual installations are required. For the following lines of code, please copy and paste each line into the terminal in the order given.

   ```bash
   pip install langchain

   pip install -U langchain-community

   pip install openai chromadb streamlit pypdf2 tiktoken nltk python-docx
   ```

4. **Set up your OpenAI API key**

   Obtain a personal API key from [OpenAI](https://openai.com/index/openai-api/) and add it to your terminal environment, like this:

   (For temporary usage, please contact the email at the very bottom.)

- **macOS / Linux**:

  ```bash
  export OPENAI_API_KEY='YOUR_OPEN_AI_KEY'
  ```

- **Windows**:

  ```bash
  set OPENAI_API_KEY='YOUR_OPEN_AI_KEY'
  ```

## **Usage**

To use Quanta Chatbot, follow these steps:

1. Ensure all dependencies are installed and OpenAI API key is incorporated into the file.
2. Run the main.py file using the following command.

   ```bash
   streamlit run main.py
   ```

3. A page will be launched on your default web browser.

   <u>**NOTICE**</u>:

   - At this stage, if the page shows a red screen with a _"ModuleNotFound"_ error, please **install** the ones listed in the error message manually at the terminal as well, using the following template. This tends to vary by computer. Then, repeat step 2.
     ```bash
     pip install <dependencies you need>
     ```
   - If _"StreamlitDuplicateElementKey"_ error pops up, just **refresh** the page once more.

4. Upload as many files as you want by dragging or selecting.
   - It is intended that files MUST BE uploaded FIRST for program to run.
5. Ask questions in English about the loaded files using the chat interface.

## **Resolving Conflicts**

While running the program, various errors might occur. If so, make sure to try these temporary solutions.

1. Try pip -> pip3 / python -> python3
2. Update the problem dependencies.

   ```bash
   pip install --upgrade <dependency being updated>
   ```

3. Ensure that your OpenAI has enough credit balance, else it won't be making successful API calls. This can be checked [here.](platform.openai.com)
4. Make sure to search up the error message that appears on the terminal window.
5. Hand type the terminal commands, instead of copying and pasting.

## **[Testing] Model Evaluation and Diagnostic Tools**

Before developing the main chatbot interface, this section shows how the internal components of the retrieval system were tested and evaluated independently. By running a series of experiments - including statistical visualizations, metric-based comparisons, and LangChain evaluations - I aim to identify the most effective configurations for semantic search and information retrieval. This includes analyzing embeddings, scoring methods like F1, NDCG, MRR, Recall@K, and evaluating the trade-offs between accuracy, speed, and memory efficiency across various approaches.

> ℹ️ **Note**  
> All files and descriptions relevant to testing can be found in the `src/testing` directory.

##

For further inquiries, please feel free to contact dongeunk@umich.edu.
