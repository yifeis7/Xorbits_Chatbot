# Use LangChain & Xorbits to Chat with customized data in an intuitive and efficient way.

In the ever-evolving world of data processing and manipulation, the need for efficient and scalable data loading tools has become paramount. The LangChain framework, known for its capability of interacting with LLMs, has taken a significant leap in this direction. With the integration of the Xorbits Document Loader, we have transformed its data loading process, making it more efficient, scalable, and versatile.

The Xorbits Document Loader is a powerful tool designed to parallelize and distribute data loading tasks. It assists LangChain in seamlessly connecting to a variety of data sources, thereby enhancing its loading capabilities and overall performance.

[Xorbits](https://doc.xorbits.io/en/latest/) is an open-source computing framework that makes it easy to scale data science and machine learning workloads — from data loading to preprocessing, tuning, training, and model serving. With its integration with LangChain, Xorbits is able to provide a seamless experience for users who want to ask questions about their own data and get results with human language prompts. This integration allows LLMs to interact with Xorbits document loader, providing users with a comprehensive solution for their data needs. With Xorbits and LangChain, users can leverage the power of LLMs to gain valuable insights from their data in a more intuitive and efficient manner.

## Xorbits Document Loader for Enhanced Data Loading

To further enhance the data loading capabilities of LangChain, the framework now leverages Xorbits Document Loader. This powerful tool enables LangChain to parallelize and distribute the loading of data, making the process even more efficient and scalable. By utilizing Xorbits Document Loader, developers can seamlessly connect LangChain to various data sources and benefit from its advanced loading capabilities.

### Pipeline Overview for Xorbits Document Loader

1. **Initialization Phase**: The loader is instantiated with a Xorbits DataFrame and the designation of the column that encapsulates the page content. During this phase, the system performs a series of checks to ensure the integrity and compatibility of the input. It verifies the type of the input data frame and check necessary dependencies.
2. **On-Demand Data Retrieval Phase**: This phase is geared towards efficient data management, particularly when dealing with vast datasets. The loader retrieves records from the DataFrame on an as-needed basis, a strategy also known as lazy loading. It constructs Document objects for each dataframe row, encapsulating the page content and the corresponding metadata. The system then yields these Document objects sequentially, thereby ensuring memory efficiency.
3. **Complete Data Retrieval Phase**: This phase is designed for scenarios where the entire DataFrame needs to be loaded simultaneously. It leverages the mechanism used in the on-demand data retrieval phase but transforms the output into a list, effectively loading the entirety of the data into memory. While this method is more straightforward, it might be less efficient when dealing with extensive datasets due to the high memory demand.

Examples:

```Python
import xorbits.pandas as pd
from langchain.document_loaders import XorbitsLoader

df = pd.read_csv("example_data/mlb_teams_2012.csv")
loader = XorbitsLoader(df, page_content_column="Team")
loader.load()
```

Output:

```
[Document(page_content='Nationals', metadata={' "Payroll (millions)"': 81.34, ' "Wins"': 98}),
 Document(page_content='Reds', metadata={' "Payroll (millions)"': 82.2, ' "Wins"': 97}),
 Document(page_content='Yankees', metadata={' "Payroll (millions)"': 197.96, ' "Wins"': 95}),
 Document(page_content='Giants', metadata={' "Payroll (millions)"': 117.62, ' "Wins"': 94}),
 Document(page_content='Braves', metadata={' "Payroll (millions)"': 83.31, ' "Wins"': 94}),
 Document(page_content='Athletics', metadata={' "Payroll (millions)"': 55.37, ' "Wins"': 94}),
 Document(page_content='Rangers', metadata={' "Payroll (millions)"': 120.51, ' "Wins"': 93}),
 Document(page_content='Orioles', metadata={' "Payroll (millions)"': 81.43, ' "Wins"': 93}),
 Document(page_content='Rays', metadata={' "Payroll (millions)"': 64.17, ' "Wins"': 90}),
 Document(page_content='Angels', metadata={' "Payroll (millions)"': 154.49, ' "Wins"': 89}),
 Document(page_content='Tigers', metadata={' "Payroll (millions)"': 132.3, ' "Wins"': 88}),
 Document(page_content='Cardinals', metadata={' "Payroll (millions)"': 110.3, ' "Wins"': 88}),
 Document(page_content='Dodgers', metadata={' "Payroll (millions)"': 95.14, ' "Wins"': 86}),
 Document(page_content='White Sox', metadata={' "Payroll (millions)"': 96.92, ' "Wins"': 85}),
 Document(page_content='Brewers', metadata={' "Payroll (millions)"': 97.65, ' "Wins"': 83}),
 Document(page_content='Phillies', metadata={' "Payroll (millions)"': 174.54, ' "Wins"': 81}),
 Document(page_content='Diamondbacks', metadata={' "Payroll (millions)"': 74.28, ' "Wins"': 81}),
 Document(page_content='Pirates', metadata={' "Payroll (millions)"': 63.43, ' "Wins"': 79}),
 Document(page_content='Padres', metadata={' "Payroll (millions)"': 55.24, ' "Wins"': 76}),
 Document(page_content='Mariners', metadata={' "Payroll (millions)"': 81.97, ' "Wins"': 75}),
 Document(page_content='Mets', metadata={' "Payroll (millions)"': 93.35, ' "Wins"': 74}),
 Document(page_content='Blue Jays', metadata={' "Payroll (millions)"': 75.48, ' "Wins"': 73}),
 Document(page_content='Royals', metadata={' "Payroll (millions)"': 60.91, ' "Wins"': 72}),
 Document(page_content='Marlins', metadata={' "Payroll (millions)"': 118.07, ' "Wins"': 69}),
 Document(page_content='Red Sox', metadata={' "Payroll (millions)"': 173.18, ' "Wins"': 69}),
 Document(page_content='Indians', metadata={' "Payroll (millions)"': 78.43, ' "Wins"': 68}),
 Document(page_content='Twins', metadata={' "Payroll (millions)"': 94.08, ' "Wins"': 66}),
 Document(page_content='Rockies', metadata={' "Payroll (millions)"': 78.06, ' "Wins"': 64}),
 Document(page_content='Cubs', metadata={' "Payroll (millions)"': 88.19, ' "Wins"': 61}),
 Document(page_content='Astros', metadata={' "Payroll (millions)"': 60.65, ' "Wins"': 55})]
```

By leveraging Xorbits Document Loader, LangChain gains significant improvements in data loading performance and scalability. This integration empowers developers to load diverse datasets into LangChain efficiently for a wide range of LLM applications.

## App: Build a chatbot with customized data using Xorbits document loader

While ChatGPT is limited to processing a maximum of 4096 tokens, our chatbot has the capability to manage large amounts of data through the use of embeddings and a vectorstore. By leveraging these advanced techniques, our chatbot can efficiently process various kinds of data through Xorbits Pandas dataframe and interact with a database to provide quick and accurate responses to user queries.

Before starting, we need to install all the dependencies. 

```
pip install streamlit streamlit_chat langchain openai faiss-cpu tiktoken xorbits
```

Here is the code snippet of our example:

```python
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.xorbits import XorbitsLoader
from langchain.vectorstores import FAISS
import tempfile
import xorbits

# Install necessary libraries
# !pip install streamlit streamlit_chat langchain openai faiss-cpu tiktoken xorbits

# Ask user for OpenAI API key and upload CSV file
user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password",
)

uploaded_file = st.sidebar.file_uploader("upload", type="csv")

# Load CSV file if uploaded
data = None
if uploaded_file:
    # Use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    df = xorbits.pandas.read_csv(tmp_file_path)
    df["All"] = df.apply(lambda row: ", ".join(str(x) for x in row), axis=1)
    loader = XorbitsLoader(df, page_content_column="All")
    data = loader.load()
# Initialize OpenAI embeddings and FAISS vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(data, embeddings)

# Create ConversationalRetrievalChain with chat model and vectorstore
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo"),
    retriever=vectorstore.as_retriever(),
)


# Function for chatbot conversation
def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state["history"]})
    st.session_state["history"].append((query, result["answer"]))
    return result["answer"]


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
            "Query:", placeholder="Talk about your data here (:", key="input"
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

```

We use streamlit to build the chatbot app with concise UI. You can query about your customized data with one click.

Here is an example:

![chat](chat.png)

## Summary

LangChain is a powerful framework that allows developers to build applications powered by language models. It enables the interaction between language models and personalized data sources, opening up new possibilities for querying and analyzing data. 