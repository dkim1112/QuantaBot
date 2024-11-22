import nltk
nltk.download("punkt")

from ui.streamlit_app import streamlit_ui

if __name__ == "__main__":
    streamlit_ui()


# TODO : Implement chatbot history.
# TODO : Allow free talking & asking general questions about the document too.
# TODO : Have options to give reference to where it got the information (through saving history)
# TODO : Add more error handling and logging
# TODO : Add more comments and docstrings
# TODO : Cut down on the time it takes for response
# TODO : Add self rag loop through invoking quanta until answer is found (RAG --> Use AI's default knowledge too.)
# TODO : Add URL abilities
# TODO : Conversation History: The quanta can access conversation history to maintain context and provide more relevant responses.
# TODO : Logical Intent Determination: It uses logic to determine user intent, ensuring accurate responses.
# TODO : UI Streamlit 건드려서 발전 시켜보기
# TODO : Implement cosine similarity logic

# DONE : Document Summarization: It can summarize documents to provide concise answers or overviews.
# DONE : Follow-up Answers: The quanta can answer follow-up questions based on previous interactions and the current conversation context.
# DONE : continuous talking (back-and-forth)

