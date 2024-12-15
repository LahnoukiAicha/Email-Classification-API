from flask import Flask, request, jsonify
from flask_cors import CORS  # For cross-origin requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Dynamically load pre-trained BERT model and tokenizer
model_dir = os.path.join(os.path.dirname(__file__), "email_classification_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = model.to(device)
model.eval()

# Solution mapping (adjust if necessary)
solution_mapping = {
    0: "Marketing Cloud", 1: "Experience Cloud", 2: "Integration Cloud", 
    3: "Sales Cloud", 4: "Analytics Cloud", 5: "Commerce Cloud", 6: "Service Cloud"
}

# Global variable to store classified emails
classified_emails_data = []

@app.route('/classify-emails', methods=['GET', 'POST'])
def classify_emails():
    global classified_emails_data
    if request.method == 'GET':
        if classified_emails_data:
            return jsonify({'status': 'success', 'classified_emails': classified_emails_data})
        return jsonify({'status': 'error', 'message': 'No classified emails available'}), 404

    if request.method == 'POST':
        try:
            # File validation
            file = request.files.get('file')
            if not file or not file.filename.endswith('.csv'):
                return jsonify({'status': 'error', 'message': 'Provide a valid CSV file'}), 400

            client_data = pd.read_csv(file, delimiter=';')
            required_columns = {'id_email', 'subject', 'email_text', 'annual_revenue', 
                                'engagement_score', 'email_opens', 'website_visits'}
            if not required_columns.issubset(client_data.columns):
                return jsonify({'status': 'error', 'message': 'Missing required columns'}), 400

            # Normalize numerical columns
            client_data['annual_revenue'] = client_data['annual_revenue'].replace('[\$,]', '', regex=True).astype(float)
            scaler = MinMaxScaler()
            for col in ['annual_revenue', 'engagement_score', 'email_opens', 'website_visits']:
                client_data[col] = scaler.fit_transform(client_data[[col]])

            # Compute Lead Score
            weights = {'engagement_score': 0.4, 'website_visits': 0.3, 'annual_revenue': 0.2, 'email_opens': 0.1}
            client_data['Lead_Score'] = sum(client_data[k] * v for k, v in weights.items())

            # Classify with BERT
            texts = client_data['subject'] + " " + client_data['email_text']
            inputs = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            client_data['Predicted_Solution'] = [solution_mapping[p] for p in predictions]
            classified_emails_data = client_data[['id_email', 'Predicted_Solution', 'Lead_Score']].to_dict(orient='records')

            return jsonify({'status': 'success', 'classified_emails': classified_emails_data})

        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
