import streamlit as st
from streamlit_chat import message
from langchain.agents import create_xorbits_agent
from langchain.llms import OpenAI
import tempfile
import xorbits


user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password",
)

uploaded_file = st.sidebar.file_uploader("upload", type="csv")
df = None
if uploaded_file:
    # Use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    df = xorbits.pandas.read_csv(tmp_file_path)
st.sidebar.write(df.head())
agent = create_xorbits_agent(OpenAI(temperature=0), df, verbose=True)


# Function for chatbot conversation
def conversational_chat(query):
    result = agent.run(query)
    st.session_state["history"].append((query, result))
    return result


# Initialize chatbot session
if "history" not in st.session_state:
    st.session_state["history"] = []

if "generated" not in st.session_state:
    st.session_state["generated"] = [
        "Hello! Ask me anything about " + uploaded_file.name
    ]

if "past" not in st.session_state:
    st.session_state["past"] = ["Hey! 👋"]

# Containers for chat history and user input
response_container = st.container()
container = st.container()

# User input and chatbot response
with container:
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_input(
            "Query:", placeholder="Ask about your data here (:", key="input"
        )
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        output = conversational_chat(user_input)
        st.session_state["past"].append(user_input)
        st.session_state["generated"].append(output)

# Display chat messages
if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state["generated"])):
            message(
                st.session_state["past"][i],
                is_user=True,
                key=str(i) + "_user",
                avatar_style="big-smile",
            )
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
