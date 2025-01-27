Flask Chatbot Project

Introduction
This project is focused on helping users gain knowledge about data science. It is a simple chatbot built using Flask, a lightweight web framework in Python. The chatbot processes user input and provides responses based on predefined content from a text file.

How It Works
1. **Text Processing:**
   - The chatbot reads data from a file called `chat.txt`.
   - It converts the text to lowercase and splits it into sentences and words.
   - Special techniques like lemmatization are used to simplify words to their base form.

2. **Greeting Detection:**
   - If the user greets the chatbot (e.g., "hello" or "hi"), it responds with a friendly message.

3. **Answering Questions:**
   - When a user asks about data science, machine learning, or deep learning, the chatbot searches for related information and provides an answer.

What You Need to Run This Project
- **Python 3.x** (Make sure Python is installed on your computer)
- **Flask** (To run the chatbot)
- **NLTK** (A library for text processing)
- **NumPy** (For numerical operations)
- **Scikit-learn** (For handling text similarity)

Steps to Set Up
1. Download the project:
   ```bash
   git clone https://github.com/yourusername/chatbot-project.git
   cd chatbot-project
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Download additional NLTK files:
   Open Python and run the following:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

How to Use
1. Start the chatbot:
   ```bash
   python image.py
   ```

2. Open the chatbot in your web browser:
   - Go to `http://127.0.0.1:5001`
   - Type your question in the input box and press enter.
   
   Example questions:
   - "What is the difference between machine learning and deep learning?"
   - "Tell me about the data science process."

Project Files Explained
```
├── image.py              # The main code to run the chatbot
├── chat.txt            # A file with chatbot responses
├── templates/          # HTML files for the web interface
└── static/             # images for styling
```
