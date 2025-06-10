import streamlit as st
import random
import os
import pandas as pd
import re
from datetime import datetime
from transformers import pipeline
from dotenv import load_dotenv

# Must be the first Streamlit command
st.set_page_config(page_title="Interview Simulator", layout="centered")

# Load environment variables
load_dotenv()

# Initialize local model pipeline (FLAN-T5-Large)
@st.cache_resource
def get_pipeline():
    return pipeline("text2text-generation", model="google/flan-t5-large")

nlp = get_pipeline()

# Function to get feedback, score, improvement, and category
def get_feedback_score_improvement(question, answer):
    prompt = (
        f"You are an AI interview coach. Please analyze the following answer.\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Provide the following in order:\n"
        f"1. Feedback: Provide clear, detailed feedback.\n"
        f"2. Suggestion: One specific improvement.\n"
        f"3. Category: Technical, Communication, or Both.\n"
        f"4. Score: X/10"
    )
    result = nlp(prompt, max_new_tokens=400)[0]["generated_text"]

    score_match = re.search(r"Score\s*[:=]\s*(\d+/10|\d+)", result, re.IGNORECASE)
    score = score_match.group(1) if score_match else "N/A"

    category_match = re.search(r"Category\s*[:=]\s*(Technical|Communication|Both)", result, re.IGNORECASE)
    category = category_match.group(1).capitalize() if category_match else "N/A"

    feedback_section = re.split(r"(?i)Score\s*[:=]", result)[0].strip()

    return feedback_section.strip(), score, category

# Function to auto-generate answer for custom question
def get_auto_answer(question):
    prompt = f"Generate a strong sample interview answer for this question:\n{question}"
    return nlp(prompt, max_new_tokens=150)[0]["generated_text"].strip()

# Save Q&A + feedback to CSV
log_path = "interview_log.csv"
if os.path.exists(log_path):
    log_df = pd.read_csv(log_path)
else:
    log_df = pd.DataFrame(columns=["timestamp", "role", "question", "answer", "feedback", "score", "category"])

st.session_state.setdefault("history", log_df)

def log_response(role, question, answer, feedback, score, category):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "role": role,
        "question": question,
        "answer": answer,
        "feedback": feedback,
        "score": score,
        "category": category
    }
    st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([log_entry])], ignore_index=True)
    st.session_state.history.to_csv(log_path, index=False)

# Role-based questions
roles = {
    "Data Analyst": [
        "Tell me about yourself.",
        "Why do you want to work in data analytics?",
        "Explain a project you’ve worked on using Python.",
        "What is the difference between inner join and left join?",
        "How do you handle missing data in a dataset?",
        "What are KPIs? Can you give an example from your experience?"
    ],
    "Data Scientist": [
        "What is overfitting in machine learning?",
        "Describe a time you tuned hyperparameters.",
        "Explain bias-variance tradeoff.",
        "What's the difference between classification and regression?",
        "When would you use logistic regression?",
        "How do you evaluate a classification model?"
    ],
    "ML Engineer": [
        "How do you deploy a model in production?",
        "Explain MLOps in simple terms.",
        "How would you scale a model?",
        "What are some challenges in model versioning?",
        "Describe the CI/CD pipeline for ML.",
        "How do you monitor model drift?"
    ]
}

# UI setup
st.title("🧠 AI Interview Simulator")
st.markdown("Practice your interview questions by role and get AI-powered feedback.")

role = st.selectbox("Select Role:", list(roles.keys()))
questions = roles[role]

custom_question = st.text_input("Or enter your own custom question:")

col1, col2 = st.columns(2)

if col1.button("📝 Get Answer"):
    if custom_question.strip():
        st.session_state["question"] = custom_question.strip()
        auto_answer = get_auto_answer(custom_question.strip())
        st.session_state["user_answer"] = auto_answer

if col2.button("🎲 Generate Question"):
    question = random.choice(questions)
    st.session_state["question"] = question
    st.session_state["user_answer"] = ""

if "question" in st.session_state:
    st.subheader("💬 Question:")
    st.write(st.session_state["question"])
    user_answer = st.text_area("Your Answer:", value=st.session_state.get("user_answer", ""), height=200, key="user_answer")

    if st.button("🧠 Get Feedback"):
        if user_answer.strip():
            feedback, score, category = get_feedback_score_improvement(st.session_state["question"], user_answer)

            st.subheader("🤖 Feedback + Score:")
            st.markdown(f"**Feedback:**\n\n{feedback}")
            st.markdown(f"**Score:** {score}")
            st.markdown(f"**Category:** {category}")

            log_response(role, st.session_state["question"], user_answer, feedback, score, category)
        else:
            st.warning("Please type your answer before requesting feedback.")

# Option to download CSV
if not st.session_state.history.empty:
    with st.expander("📜 Session History"):
        st.dataframe(st.session_state.history.tail(50), use_container_width=True)

    csv = st.session_state.history.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Session Log (CSV)", csv, file_name="interview_log.csv")

st.markdown("---")
st.markdown("Built by Minal Pawar | Local LLM: FLAN-T5")
