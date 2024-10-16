from sentence_transformers import SentenceTransformer
import numpy as np
import openai

# Set your OpenAI API key
# openai.api_key = "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"

# Load a pre-trained model for Japanese embeddings
model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')




# Specify the path to your text file
file_path = "words.txt"
medical_terms = []
# Open the file and read it line by line
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        # Remove any leading/trailing whitespace (like newlines)
        line = line.strip()
        # print(line[4:])  # Process the line as needed
        for item in line[4:].split(", "):
            item = item.replace(' and', '')
            # print(item)
            medical_terms.append(item)


# Create embeddings for the medical terms
medical_term_embeddings = model.encode(medical_terms)



# Specify the path to your text file
file_path = "original_transcript.txt"

# Open the file and read its content
with open(file_path, "r", encoding="utf-8") as file:
    # Read the entire content
    content = file.read()

# Remove all line breaks (newlines)
conversation_text = content.replace('\n', '')

# Optionally, print the result
# print(conversation_text)





# Split the transcription into words
transcribed_words = conversation_text.split()

# Function to find the closest medical term using embeddings
def find_closest_term(word, medical_terms, medical_term_embeddings):
    # Create an embedding for the word
    word_embedding = model.encode([word])
    
    # Compute cosine similarities between the word and medical terms
    similarities = np.dot(medical_term_embeddings, word_embedding.T).flatten()
    
    # Find the index of the most similar medical term
    closest_idx = np.argmax(similarities)
    
    # If the similarity is high enough, return the closest medical term
    if similarities[closest_idx] > 0.99:  # You can adjust the similarity threshold
        return medical_terms[closest_idx]
    else:
        return word  # Return the original word if no close match is found

# Correct the transcription by replacing words with closest medical terms
corrected_transcription = []
for word in transcribed_words:
    closest_term = find_closest_term(word, medical_terms, medical_term_embeddings)
    corrected_transcription.append(closest_term)

# Join the corrected words into a final corrected transcription
corrected_conversation = " ".join(corrected_transcription)
print("Corrected Transcription:", corrected_conversation)




          

