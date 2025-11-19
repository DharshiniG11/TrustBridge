This project is an end-to-end ML pipeline dashboard built using Streamlit, connected with Supabase, and designed to manage milestones and run automated pipelines.

🔧 Features
Add, view, and manage project milestones

Run the ML pipeline with one click

Store and fetch data from Supabase

Clean and simple Streamlit UI

📂 Project Structure
bash
Copy code
TrustBridge/
│── app.py               # Streamlit dashboard
│── ml_pipeline.py       # Main ML pipeline
│── supabase_client.py   # Supabase connection
│── requirements.txt
│── .streamlit/
│     └── secrets.toml   # API keys



🚀 Setup Instructions
1. Clone the Repository
bash
Copy code
git clone https://github.com/YOUR-USERNAME/TrustBridge.git
cd TrustBridge


3. Install Dependencies
bash
Copy code
pip install -r requirements.txt


5. Add Secrets
Create a file:

bash
Copy code
.streamlit/secrets.toml
Add:

toml
Copy code
SUPABASE_URL = "your-url"
SUPABASE_KEY = "your-key"


4. Run the App
bash
Copy code
streamlit run app.py

6. Run Pipeline Manually
bash
Copy code
python ml_pipeline.py


🧪 Tech Stack
Python
Streamlit
Supabase
PostgreSQL
