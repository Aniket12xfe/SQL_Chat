import streamlit as st
import mysql.connector
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.sql_database import SQLDatabase
from urllib.parse import quote_plus
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Load environment variables
load_dotenv()
api = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api)

# Function to establish connection with MYSQL database
def connect_database(hostname: str, port: str, username: str, password: str, database: str) -> SQLDatabase:
    encoded_password = quote_plus(password)
    db_uri = f"mysql+mysqlconnector://{username}:{encoded_password}@{hostname}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

# Wrapping the model call in a RunnableLambda to handle compatibility
def get_sql_chain(db):
    prompt_template = """
        You are a senior data analyst.
        Based on the table schema provided below, write a SQL query that answers the question.
        Consider the conversation history.

        <SCHEMA> {schema} </SCHEMA>

        Conversation History: {conversation_history}

        Write only the SQL query without any additional text.

        For example:
        Question: Who are the top 3 artists with the most tracks?
        SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;

        Response Format:
            Question: {question}
            SQL Query:
    """
    
    # Define the model as a runnable lambda function
    model = RunnableLambda(lambda x: ChatGoogleGenerativeAI("gemini-1.5-flash").predict(x))

    prompt = ChatPromptTemplate.from_template(template=prompt_template)

    # Function to return schema details of the database
    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | model
        | StrOutputParser()
    )

# Function to convert SQL query into Natural Language response
def get_response(user_query: str, db: SQLDatabase, conversation_history: list):
    sql_chain = get_sql_chain(db)

    response_template = """
        You are a senior data analyst. 
        Given the database schema details, question, SQL query, and SQL response, 
        write a natural language response for the SQL query.

        <SCHEMA> {schema} </SCHEMA>

        Conversation History: {conversation_history}
        SQL Query: <SQL> {sql_query} </SQL>
        Question: {question}
        SQL Response: {response}

        Response Format:
            SQL Query:
            Natural Language Response:
    """
    
    prompt = ChatPromptTemplate.from_template(template=response_template)
    
    model = RunnableLambda(lambda x: ChatGoogleGenerativeAI("gemini-1.5-flash").predict(x))

    chain = (
        RunnablePassthrough.assign(sql_query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["sql_query"])
        )
        | prompt
        | model
        | StrOutputParser()
    )

    return chain.invoke({
        "question": user_query,
        "conversation_history": conversation_history
    })

# Initialize conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [
        AIMessage(content="Hello! I am a SQL assistant. Ask me questions about your MYSQL database.")
    ]

# Page config
st.set_page_config(page_title="SQL Chat", page_icon=":speech_balloon:")
st.title("SQL Chat")

# Sidebar for database connection
with st.sidebar:
    st.subheader("Settings")
    st.write("Connect your MYSQL database and chat with it!")

    # Connect to database
    st.text_input("Hostname", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("Username", value="root", key="Username")
    st.text_input("Password", type="password", key="Password")
    st.text_input("Database", key="Database")

    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            try:
                db = connect_database(
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["Username"],
                    st.session_state["Password"],
                    st.session_state["Database"]
                )

                st.session_state.db = db
                st.success("Connected to Database!")

            except mysql.connector.Error as err:
                st.error(f"Error connecting to database: {err}")

# Interactive chat interface
for message in st.session_state.conversation_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)

    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# User Query
user_query = st.chat_input("Question your database...")

if user_query and st.session_state.get("db"):
    st.session_state.conversation_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.conversation_history)
        
        if response:  # Check if a valid response is returned
            st.markdown(response)
            st.session_state.conversation_history.append(AIMessage(content=response))
        else:
            fallback_message = "I'm sorry, I couldn't process that request. Could you please try rephrasing or asking something else?"
            st.markdown(fallback_message)
            st.session_state.conversation_history.append(AIMessage(content=fallback_message))
