from transformers import pipeline

class NLPModuleClassifier:
    def __init__(self):
        # We initialize a zero-shot classification pipeline. 
        # bart-large-mnli is a standard, highly accurate model for this.
        print("Loading NLP Model...")
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # Define the buckets you want to categorize modules into. 
        # You can change these to "whatever" you need at any time.
        self.candidate_labels = [
            "security and authentication",
            "database operations",
            "user interface",
            "networking and api",
            "data processing",
            "utility functions"
        ]

    def identify_module(self, extracted_text):
        """
        extracted_text: A string of aggregated text from the module 
        (e.g., "Handles JWT validation, password hashing, and user sessions.")
        """
        if not extracted_text.strip():
            return "unknown"

        # The model will score the extracted text against your candidate labels
        result = self.classifier(extracted_text, self.candidate_labels)
        
        # Get the top matching category and its confidence score
        top_category = result['labels'][0]
        confidence = result['scores'][0]
        
        return {
            "predicted_category": top_category,
            "confidence_score": round(confidence, 4),
            "all_scores": dict(zip(result['labels'], result['scores']))
        }

# --- Example Usage ---
# nlp_classifier = NLPModuleClassifier()
# docstrings_from_ast = "This module verifies user tokens, prevents SQL injection, and hashes passwords."
# classification = nlp_classifier.identify_module(docstrings_from_ast)
# print(classification['predicted_category'])  # Output: 'security and authentication'