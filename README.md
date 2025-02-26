# SMS Spam Classifier

This is a simple **SMS Spam Classifier** that detects whether a given message is **spam or ham (not spam)** using a machine learning model.

## 🚀 Features
- Classifies SMS messages as **Spam** or **Ham (Not Spam)**.
- Uses a trained model to predict message categories.
- Includes a basic web interface (`index.html`) for user input.

## 📁 Project Structure
```plaintext
SMS-Spam-Classifier/
│── templates/
│   ├── index.html       # Web interface for classification
│── sms_predict.py            # Python script for processing and classification
│── spam_data.txt       # Sample dataset or messages for testing
│── README.md           # Project documentation
|── requirements.txt    # Necessary imports for application
```
## 🛠️ Usage
### :one: Install Python (if not already installed)
- **Windows**: Download and install from `python.org`
- **MacOS/ Linux**: Python is usually pre-installed. Check by running:
  ```
  python --version
  ```
  If not installed, use:
  - `sudo apt install python3` for Debian/ Ubuntu
  - `brew install python` for MacOS (using Homebrew)
### :two: Create and activate a virtual environment:
Navigate to the project folder and run:
```
python -m venv venv
```
Then, run:
```
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```
### :three: Install the required dependencies:
Upon activation of the virtual environment, run:
```
pip install -r requirements.txt
```
### :four: Run the Flask app:
Finally, run:
```
flask --app sms_predict run
```
### :five: Open in a browser
Go to:
`http://127.0.0.1:5000/`
and you'll be able to enter messages and have them classified as Spam or Not Spam!
