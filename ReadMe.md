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

   pip install openai chromadb streamlit pypdf2 tiktoken nltk
   ```

4. **Set up your OpenAI API key**

   Obtain a personal API key from OpenAI and add it to your terminal environment, like this:

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

   **NOTICE**: At this stage, if the page shows a red screen with a "module not found" error, please install the ones listed in the error message manually at the terminal as well, using the following template. This tends to vary by computer.

   ```bash
   pip install <dependencies you need>
   ```

4. Upload as many files as you want by dragging or selecting.
5. Ask questions in English about the loaded files using the chat interface.

##

For further inquiry, please feel free to contact dongeunk@umich.edu.
