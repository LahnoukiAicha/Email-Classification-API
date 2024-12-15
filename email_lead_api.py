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
model_name = "alpha2002/model"  # Replace with your Hugging Face repository
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
model = BertForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = BertTokenizer.from_pretrained(model_name)
model.eval()
print("Model loaded successfully!")

# Solution mapping (adjust if necessary)
solution_mapping = {
    0: "Marketing Cloud", 1: "Experience Cloud", 2: "Integration Cloud",
    3: "Sales Cloud", 4: "Analytics Cloud", 5: "Commerce Cloud", 6: "Service Cloud"
}

# Global variable to store classified emails
classified_emails_data = []

@app.route('/classify-emails', methods=['POST'])
def classify_emails():
    global classified_emails_data
    try:
        # Check if JSON or File input
        if request.is_json:
            input_data = request.get_json()
            client_data = pd.DataFrame(input_data)
        else:
            file = request.files.get('file')
            if not file or not file.filename.endswith('.csv'):
                return jsonify({'status': 'error', 'message': 'Invalid or missing CSV file'}), 400
            client_data = pd.read_csv(file, delimiter=';')

        # Check required columns
        required_columns = {'id_email', 'subject', 'email_text', 'annual_revenue',
                            'engagement_score', 'email_opens', 'website_visits'}
        if not required_columns.issubset(client_data.columns):
            return jsonify({'status': 'error', 'message': 'Missing required columns'}), 400

        # Clean annual revenue column
        client_data['annual_revenue'] = client_data['annual_revenue'].replace('[\$,]', '', regex=True).replace(',', '', regex=True).astype(float)

        # Columns to normalize
        columns_to_normalize = ['annual_revenue', 'engagement_score', 'email_opens', 'website_visits']

        # Handle single record normalization manually
        if len(client_data) == 1:
            print("Single record detected. Performing manual normalization.")
            max_values = {'annual_revenue': 1e9, 'engagement_score': 1, 'email_opens': 100, 'website_visits': 100}  # Reasonable maximums
            for col in columns_to_normalize:
                client_data[col] = client_data[col] / max_values[col]
        else:
            # Apply MinMaxScaler for multiple records
            scaler = MinMaxScaler()
            client_data[columns_to_normalize] = scaler.fit_transform(client_data[columns_to_normalize])

        # Compute Lead Score using weights
        weights = {'engagement_score': 0.4, 'website_visits': 0.3, 'annual_revenue': 0.2, 'email_opens': 0.1}
        client_data['Lead_Score'] = (
            client_data['engagement_score'] * weights['engagement_score'] +
            client_data['website_visits'] * weights['website_visits'] +
            client_data['annual_revenue'] * weights['annual_revenue'] +
            client_data['email_opens'] * weights['email_opens']
        )

        # Prepare text for classification
        texts = client_data['subject'] + " " + client_data['email_text']
        inputs = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)

        # Predict solutions using the BERT model
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        # Map predictions to solutions
        client_data['Predicted_Solution'] = [solution_mapping[p] for p in predictions]
        classified_emails_data = client_data[['id_email', 'Predicted_Solution', 'Lead_Score']].to_dict(orient='records')

        # Return JSON response
        return jsonify({'status': 'success', 'classified_emails': classified_emails_data})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
