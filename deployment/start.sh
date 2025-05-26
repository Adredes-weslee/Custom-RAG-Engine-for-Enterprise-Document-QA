    #!/bin/bash
# Start the Ollama service in the background
/bin/ollama serve &
# Navigate to the visiera directory and run the Streamlit app
cd visiera
streamlit run src/main.py