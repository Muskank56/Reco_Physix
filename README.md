RecoPhysix 🔭📘

RecoPhysix is an AI-powered quiz recommendation engine designed for NEET and JEE aspirants focusing on Physics.
It analyzes student performance and recommends personalized quizzes based on topic mastery and difficulty level.

🔍 Features

Topic-wise classification of questions

Adaptive quiz recommendations

Performance feedback system

Trained using Random Forest Classifier

Streamlit UI for interactive quiz attempts

🎯 Technologies Used

Python (Pandas, Scikit-learn, NumPy)

Jupyter Notebook / Google Colab for model training

Streamlit for UI

Git & GitHub for version control

🧠 How It Works

Input quiz data: score, attempts, time taken, topic, difficulty

Preprocessing with OneHot/Label Encoding

Trains ML model to predict the next best topic

Recommends quizzes adaptively (30 questions per topic)

Students attempt quizzes via Streamlit app and get real-time recommendations

📁 Project Structure

RecoPhysix.ipynb → Main project notebook (training & preprocessing)

app.py → Streamlit UI for quiz recommendation

quiz_recommender.pkl → Trained ML model

topic_quizzes/ → Folder containing topic-wise quiz files (CSV)

requirements.txt → Python dependencies

README.md → Project overview

🚀 Future Work

Expand to Chemistry, Biology, and Mathematics

Improve recommendation using deep learning

Add leaderboards & progress tracking for students

🔹 Requirements

Install dependencies:

pip install -r requirements.txt


Dataset used: RecoPhysix_preprocessed_dataset.csv

📌 Author

👩‍💻 Muskan Kanaujia
🔗 GitHub
