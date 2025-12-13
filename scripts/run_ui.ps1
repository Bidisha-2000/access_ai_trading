python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
streamlit run src/omago_ai/ui/app.py
