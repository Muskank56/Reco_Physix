import os 
import time
import random
import pandas as pd
import streamlit as st
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import datetime
import numpy as np

# ------------------------
# Config / Paths
# ------------------------
QUIZ_DIR = "topic_quizzes"
HISTORY_FILE = "student_performance.csv"
MODEL_PATH = "quiz_recommender.pkl"
#ENCODER_PATH = "one_hot_encoder.joblib"
ALL_TOPICS = [
    "Electricity", "Kinematics", "Mechanics",
    "Mixed Questions", "Modern Physics", "Optics", "Thermodynamics"
]

# ------------------------
# Utilities
# ------------------------
def ensure_history_file():
    if not os.path.exists(HISTORY_FILE):
        cols = ["student_id", "name", "topic", "score", "time_taken", "attempts", "recommended_next_topic", "timestamp"]
        pd.DataFrame(columns=cols).to_csv(HISTORY_FILE, index=False)

def next_student_id():
    """Generate next ID like S001, S002... based on HISTORY_FILE contents."""
    ensure_history_file()
    df = pd.read_csv(HISTORY_FILE)
    if df.empty:
        return "S001"
    seen = df["student_id"].dropna().unique().tolist()
    max_num = 0
    for sid in seen:
        try:
            max_num = max(max_num, int(str(sid).replace("S", "").strip()))
        except Exception:
            continue
    return f"S{max_num+1:03d}"

def topic_to_quiz_path(topic: str) -> str:
    return os.path.join(QUIZ_DIR, f"{topic.replace(' ', '_')}_quiz.csv")

def load_quiz(topic: str) -> pd.DataFrame:
    path = topic_to_quiz_path(topic)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Quiz file not found for topic '{topic}': {path}")
    dfq = pd.read_csv(path)
    required_cols = ["question", "option_1", "option_2", "option_3", "option_4", "correct_answer"]
    missing = [c for c in required_cols if c not in dfq.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return dfq

def standardize_answer(value: str) -> str:
    """Normalize answer to the literal option text to compare reliably."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()

# ------------------------
# Model: load or (re)train
# ------------------------
@st.cache_resource
def load_or_train_model():
    """
    1) Try to load model + encoder from disk.
    2) Else, if HISTORY_FILE has data (>=100 rows), train a model and save it.
    3) Else, return a tiny fallback that always recommends 'Mixed Questions'.
    """
    # Try loading saved artifacts
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        return model, None, True   # model, encoder (None if not saved separately), status
    else:
        return None, None, False
          

    # Else, try to train from history
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE,on_bad_lines="skip", engine="python")
        needed = {"topic", "score", "time_taken", "attempts", "recommended_next_topic"}
        if len(df) >= 100 and needed.issubset(set(df.columns)):
            # Prepare features
            X = df.drop(columns=["recommended_next_topic", "student_id", "name", "timestamp"], errors="ignore")
            y = df["recommended_next_topic"].fillna("Mixed Questions")

            # Robust OneHotEncoder creation to handle different sklearn versions
            try:
                enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                enc = OneHotEncoder(handle_unknown="ignore", sparse=False)

            topic_encoded = enc.fit_transform(X[["topic"]])
            # If output is sparse, convert to array
            if hasattr(topic_encoded, "toarray"):
                topic_encoded = topic_encoded.toarray()
            topic_df = pd.DataFrame(topic_encoded, columns=enc.get_feature_names_out(["topic"]))
            X_final = pd.concat([X.drop(columns=["topic"]).reset_index(drop=True), topic_df], axis=1)

            # Train RandomForest
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X_final, y)

            # Save artifacts
            joblib.dump(rf, MODEL_PATH)
            joblib.dump(enc, ENCODER_PATH)
            return rf, enc, True

    # Fallback model
    class FallbackModel:
        feature_names_in_ = None
        def predict(self, X):
            # Accept DataFrame or array-like
            n = len(X)
            return ["Mixed Questions"] * n

    # Fit encoder on known topics so transform works at predict time
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    enc.fit(pd.DataFrame(ALL_TOPICS, columns=["topic"]))
    return FallbackModel(), enc, False

model, encoder, model_ok = load_or_train_model()

# ------------------------
# Prediction & Logging
# ------------------------
def predict_next_topic(score: int, time_taken: int, attempts: int, topic: str) -> str:
    """Match model input column order to avoid warnings and ensure correctness."""
    # Encode topic properly (use DataFrame with column name)
    topic_encoded = encoder.transform(pd.DataFrame([[topic]], columns=["topic"]))
    if hasattr(topic_encoded, "toarray"):
        topic_encoded = topic_encoded.toarray()
    topic_df = pd.DataFrame(topic_encoded, columns=encoder.get_feature_names_out(["topic"]))

    row = pd.DataFrame([{"score": score, "time_taken": time_taken, "attempts": attempts}])
    row_final = pd.concat([row, topic_df], axis=1)

    # If the underlying model exposes feature_names_in_, reindex to that order
    if getattr(model, "feature_names_in_", None) is not None:
        row_final = row_final.reindex(columns=model.feature_names_in_, fill_value=0)

    # 🔍 Debug logging (you can comment these out later)
    st.write("🔍 Prediction Input Features:", row_final)
    try:
        pred = model.predict(row_final)[0]
        st.write("🔍 Model Predicted:", pred)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        pred = "Mixed Questions"

    return pred

def log_attempt(student_id, name, topic, score, time_taken, attempts, recommended_next_topic):
    ensure_history_file()
    df = pd.read_csv(HISTORY_FILE)
    entry = {
        "student_id": student_id,
        "name": name,
        "topic": topic,
        "score": score,
        "time_taken": time_taken,
        "attempts": attempts,
        "recommended_next_topic": recommended_next_topic,
        "timestamp": pd.Timestamp.utcnow().isoformat()
    }
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)

def get_attempt_count(student_id: str, topic: str) -> int:
    ensure_history_file()
    df = pd.read_csv(HISTORY_FILE)
    if df.empty:
        return 0
    if "student_id" not in df.columns or "topic" not in df.columns or "attempts" not in df.columns:
        return 0
    filt = (df["student_id"] == student_id) & (df["topic"] == topic)
    if filt.any():
        try:
            return int(df.loc[filt, "attempts"].max())
        except Exception:
            return 0
    return 0

def show_attempt_history(student_id: str):
    ensure_history_file()
    df = pd.read_csv(HISTORY_FILE)
    if df.empty:
        st.info("No attempts logged yet.")
        return

    # Safely handle missing timestamp column (older files)
    if "timestamp" in df.columns:
        try:
            sub = df[df["student_id"] == student_id].sort_values("timestamp", ascending=False)
        except Exception:
            # if timestamp format is inconsistent, fallback to unsorted
            st.warning("Could not sort by timestamp due to format; showing unsorted history.")
            sub = df[df["student_id"] == student_id]
    else:
        st.warning("History file missing 'timestamp' column (old records). New attempts will include it.")
        sub = df[df["student_id"] == student_id]

    if sub.empty:
        st.info("No attempts found for this Student ID yet.")
    else:
        st.dataframe(sub.reset_index(drop=True))

# ------------------------
# Session State Helpers
# ------------------------
def init_state():
    st.session_state.setdefault("student_id", "")
    st.session_state.setdefault("name", "")
    st.session_state.setdefault("page", "home")
    st.session_state.setdefault("current_topic", ALL_TOPICS[0])
    st.session_state.setdefault("quiz_started", False)
    st.session_state.setdefault("quiz_start_time", None)
    st.session_state.setdefault("quiz_df", None)
    st.session_state.setdefault("answers", {})
    st.session_state.setdefault("last_recommendation", None)
    st.session_state.setdefault("last_score", None)
    st.session_state.setdefault("last_time_taken", None)

init_state()

# ------------------------
# UI: Sidebar (Login)
# ------------------------
with st.sidebar:
    st.header("Student Login / Signup")
    existing_id = st.text_input("Enter Student ID (if you have one)", value=st.session_state["student_id"])
    name = st.text_input("Your Name (optional)", value=st.session_state["name"])

    col_a, col_b = st.columns(2)
    if col_a.button("Use ID"):
        if existing_id.strip():
            st.session_state["student_id"] = existing_id.strip()
            st.session_state["name"] = name.strip()
            st.success(f"Welcome back, {st.session_state['student_id']}")
        else:
            st.warning("Please enter a Student ID or generate a new one.")

    if col_b.button("Generate New ID"):
        st.session_state["student_id"] = next_student_id()
        st.session_state["name"] = name.strip()
        st.success(f"Your new ID is **{st.session_state['student_id']}** — please save it for future logins.")

    st.divider()
    if st.session_state["student_id"]:
        st.caption("Your attempt history:")
        show_attempt_history(st.session_state["student_id"])

# ------------------------
# UI: Pages
# ------------------------
st.title("RecoPhysix • Personalized Physics Quizzes")

# ---- Home Page ----
if st.session_state["page"] == "home":
    st.subheader("Start a Quiz")
    if not st.session_state["student_id"]:
        st.warning("Please enter or generate a Student ID in the sidebar to begin.")
    else:
        topic = st.selectbox("Choose a topic", ALL_TOPICS, index=ALL_TOPICS.index(st.session_state["current_topic"]))
        st.session_state["current_topic"] = topic

        
        quiz_file = os.path.join(QUIZ_DIR, f"{topic}_quiz.csv")
        num_q = 30
        quiz_df = pd.read_csv(quiz_file, on_bad_lines="skip",engine="python")
        quiz_df = quiz_df.sample(n=num_q, random_state=None).reset_index(drop=True)

        if st.button("Start Quiz"):
            # Load quiz and sample questions
            dfq = load_quiz(topic)
            if len(dfq) > num_q:
                dfq = dfq.sample(n=num_q, random_state=random.randint(0, 10_000)).reset_index(drop=True)
            else:
                dfq = dfq.reset_index(drop=True)

            st.session_state["quiz_df"] = dfq
            st.session_state["answers"] = {}
            st.session_state["quiz_started"] = True
            st.session_state["quiz_start_time"] = time.time()
            st.session_state["page"] = "quiz"

# ---- Quiz Page ----
if st.session_state["page"] == "quiz":
    st.subheader(f"Quiz • {st.session_state['current_topic']}")
    if not st.session_state["quiz_started"] or st.session_state["quiz_df"] is None:
        st.info("No quiz in progress. Go to Home and click **Start Quiz**.")
        if st.button("Back to Home"):
            st.session_state["page"] = "home"
    else:
        # Render questions
        dfq = st.session_state["quiz_df"]
        for i, row in dfq.iterrows():
            with st.expander(f"Q{i+1}. {row['question']}", expanded=True):
                key = f"q_{i}"
                options = [row["option_1"], row["option_2"], row["option_3"], row["option_4"]]
                # Render radio and keep answer in session state via the widget itself
                st.session_state["answers"][key] = st.radio(
                    "Select one:",
                    options,
                    key=key,
                    index=None,
                )

        col1, col2 = st.columns(2)
        if col1.button("End Quiz & Submit"):
            # Calculate score
            score = 0
            for i, row in dfq.iterrows():
                key = f"q_{i}"
                chosen = standardize_answer(st.session_state["answers"].get(key, ""))
                correct = standardize_answer(row["correct_answer"])
                if chosen and correct and chosen == correct:
                    score += 1

            elapsed = int(time.time() - st.session_state["quiz_start_time"])
            topic = st.session_state["current_topic"]

            # Attempt # for this student + topic
            attempts_so_far = get_attempt_count(st.session_state["student_id"], topic)
            attempts = attempts_so_far + 1

            # Predict next topic
            try:
                next_topic = predict_next_topic(score=score, time_taken=elapsed, attempts=attempts, topic=topic)
            except Exception as e:
                next_topic = "Mixed Questions"
                st.warning(f"Prediction fallback used: {e}")

            # Log to history (timestamp added inside function)
            log_attempt(
                student_id=st.session_state["student_id"],
                name=st.session_state["name"],
                topic=topic,
                score=score,
                time_taken=elapsed,
                attempts=attempts,
                recommended_next_topic=next_topic
            )

            st.success(f"✅ Quiz submitted! Score: **{score}/{len(dfq)}** • Time: **{elapsed} sec**")
            st.info(f"🎯 Recommended Next Topic: **{next_topic}**")

            st.session_state["last_score"] = score
            st.session_state["last_time_taken"] = elapsed
            st.session_state["last_recommendation"] = next_topic

            # Reset quiz state and move to result page
            st.session_state["quiz_started"] = False
            st.session_state["quiz_df"] = None
            st.session_state["quiz_start_time"] = None
            st.session_state["page"] = "result"

        if col2.button("Cancel & Back"):
            st.session_state["quiz_started"] = False
            st.session_state["quiz_df"] = None
            st.session_state["quiz_start_time"] = None
            st.session_state["page"] = "home"

# ---- Result Page ----
if st.session_state["page"] == "result":
    st.subheader("Your Result")
    if st.session_state["last_score"] is None:
        st.info("No recent result. Start a quiz from Home.")
    else:
        st.write(f"**Score:** {st.session_state['last_score']}  \n"
                 f"**Time Taken:** {st.session_state['last_time_taken']} sec  \n"
                 f"**Recommended Next Topic:** {st.session_state['last_recommendation']}")
        next_topic = st.session_state["last_recommendation"]

        coln1, coln2 = st.columns(2)
        if coln1.button(f"Start Next Quiz: {next_topic}"):
            # Jump straight into next topic quiz
            try:
                dfq = load_quiz(next_topic)
                dfq = dfq.sample(n=min(10, len(dfq)), random_state=random.randint(0, 10_000)).reset_index(drop=True)
                st.session_state["current_topic"] = next_topic
                st.session_state["quiz_df"] = dfq
                st.session_state["answers"] = {}
                st.session_state["quiz_started"] = True
                st.session_state["quiz_start_time"] = time.time()
                st.session_state["page"] = "quiz"
            except Exception as e:
                st.error(str(e))

        if coln2.button("Back to Home"):
            st.session_state["page"] = "home"

# Footer: model status
st.divider()
if model_ok:
    st.caption("Model: loaded ✓")
else:
    st.caption("Model: fallback (train later from student_performance.csv)")