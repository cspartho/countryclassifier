import gradio as gr
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from sklearn.preprocessing import LabelEncoder
import numpy as np
# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained("./models")
tokenizer = BertTokenizer.from_pretrained("./models")

# Initialize the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['au', 'be', 'br', 'ca', 'es', 'fr', 'jp', 'mx', 'us', 'za'])  # Example; replace with your actual classes


country_names = {
    'au': 'Australia',
    'be': 'Belgium',
    'br': 'Brazil',
    'ca': 'Canada',
    'es': 'Spain',
    'fr': 'France',
    'jp': 'Japan',
    'mx': 'Mexico',
    'us': 'United States',
    'za': 'South Africa'
}
def predict_country(address):
    inputs = tokenizer(address, padding=True, truncation=True, return_tensors="pt")
    
    model.eval()
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    
    predicted_country_code = label_encoder.inverse_transform(predictions.cpu().numpy())[0]
    predicted_country_name = country_names.get(predicted_country_code, "Unknown")
    
    return predicted_country_name

interface = gr.Interface(fn=predict_country, 
                         inputs=gr.Textbox(label="Enter Address"), 
                         outputs=gr.Textbox(label="Predicted Country"))

interface.launch(share=True)
