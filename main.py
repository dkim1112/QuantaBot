from ui.streamlit_app import streamlit_ui
import nltk
nltk.download("punkt")

if __name__ == "__main__":
    streamlit_ui()

# TODO : Have options to give reference to where it got the information (through saving history)
# TODO : Cut down on the time it takes for response
# TODO : Add self rag loop through invoking quanta until answer is found (RAG --> Use AI's default knowledge too. --> Questions might not be answered in the document, but general knowledge.)
# TODO : Resolve issue of extracting info from previous file, which is inaccurate.
# TODO : Add URL abilities
# TODO : Conversation History: The quanta can access conversation history to maintain context and provide more relevant responses.
# TODO : Logical Intent Determination: It uses logic to determine user intent, ensuring accurate responses.
# TODO : UI Streamlit Revision
# TODO : Implement cosine similarity logic, reduction techniques, and more.

# DONE : Document Summarization: It can summarize documents to provide concise answers or overviews.
# DONE : Follow-up Answers: The quanta can answer follow-up questions based on previous interactions and the current conversation context.
# DONE : Continuous talking (back-and-forth)
# DONE : Allow free talking & asking general questions about the document too.


