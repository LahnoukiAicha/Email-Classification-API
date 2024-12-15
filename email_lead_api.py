from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load pre-trained model and tokenizer from Hugging Face
model_name = "alpha2002/model"  # Use your Hugging Face repository name
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = BertTokenizer.from_pretrained(model_name)
model.eval()

# Solution mapping (adjust if necessary)
solution_mapping = {
    0: "Marketing Cloud", 1: "Experience Cloud", 2: "Integration Cloud", 
    3: "Sales Cloud", 4: "Analytics Cloud", 5: "Commerce Cloud", 6: "Service Cloud"
}

classified_emails_data = []

@app.route('/classify-emails', methods=['POST'])
def classify_emails():
    global classified_emails_data
    try:
        # Check for file input
        file = request.files.get('file')
        if not file or not file.filename.endswith('.csv'):
            return jsonify({'status': 'error', 'message': 'Invalid or missing CSV file'}), 400
        
        # Load and validate file
        client_data = pd.read_csv(file, delimiter=';')
        required_columns = {'id_email', 'subject', 'email_text', 'annual_revenue', 
                            'engagement_score', 'email_opens', 'website_visits'}
        if not required_columns.issubset(client_data.columns):
            return jsonify({'status': 'error', 'message': 'Missing required columns'}), 400
        
        # Normalize numerical columns
        scaler = MinMaxScaler()
        for col in ['annual_revenue', 'engagement_score', 'email_opens', 'website_visits']:
            client_data[col] = scaler.fit_transform(client_data[[col]])

        # Compute Lead Score
        weights = {'engagement_score': 0.4, 'website_visits': 0.3, 'annual_revenue': 0.2, 'email_opens': 0.1}
        client_data['Lead_Score'] = sum(client_data[k] * v for k, v in weights.items())

        # Prepare text for classification
        texts = client_data['subject'] + " " + client_data['email_text']
        inputs = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)

        # Predict solutions
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
