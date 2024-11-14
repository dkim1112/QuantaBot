import nltk
nltk.download("punkt")

from ui.streamlit_app import streamlit_ui

if __name__ == "__main__":
    streamlit_ui()


# TODO : continuous talking (back-and-forth)
# TODO : Implement chatbot history.
# TODO : Allow free talking & asking general questions about the document too.
# TODO : Have options to give reference to where it got the information (through saving history).
# TODO : Cut document at first when received, to only relevant infos?

# TODO : Add more error handling and logging
# TODO : Add more comments and docstrings
# TODO : Cut down on the time it takes for response
# TODO : Add streaming responses
# TODO : Add self rag loop through invoking quanta until answer is found
# TODO : Add URL abilities
# TODO : Conversation History: The quanta can access conversation history to maintain context and provide more relevant responses.
# TODO : Document Summarization: It can summarize documents to provide concise answers or overviews.
# TODO : Follow-up Answers: The quanta can answer follow-up questions based on previous interactions and the current conversation context.
# TODO : Logical Intent Determination: It uses logic to determine user intent, ensuring accurate responses.
# TODO : UI Streamlit 건드려서 발전 시켜보기