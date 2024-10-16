import openai

# Your OpenAI API token
openai.api_key = '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'  # Replace with your actual API key 


# Step 1: Transcribe the conversation using Whisper
# audio_file_path = "audio1642696603.wav"  # Replace with the path to your audio file

# with open(audio_file_path, "rb") as audio_file:
#     # Call the Whisper model for transcription
#     transcription_response = openai.Audio.transcribe(
#         model="whisper-1",
#         file=audio_file,
#         language="ja"  # Specify the language (for Japanese)
#     )

# # Extracting the transcription
# conversation_text = transcription_response['text']

file_path = "original_transcript.txt"

# Open the file and read its content
with open(file_path, "r", encoding="utf-8") as file:
    # Read the entire content
    content = file.read()

# Remove all line breaks (newlines)
conversation_text = content.replace('\n', '')

medical_words = ['バセドー病', 'クラックル音', 'チラージン']

# soap_prompt = f"""
# 次の日本語の会話は元の音声から転写されており、不完全で誤りが含まれている可能性があります。適切な医学用語に修正し、以下の医学用語リストにある言葉を必要に応じて使用してください。リスト内の言葉と意味が似ている場合は、それを優先して使用してください。また、要約では主語（例: 医師、患者、彼など）を省略し、文脈に応じて簡潔に記述してください。

# 医学用語リスト: {', '.join(medical_words)}

# 以下の構成で要約を提供してください。すべて日本語で記述してください。

# {conversation_text}

# 要約構成:
# S: [主観的情報] (1行で十分です) （「患者」や「医者」などの主語を使用せず、「彼」「彼女」などの代名詞を使用し、文脈に応じて省略形を使ってください）
# O: [客観的情報] （箇条書きで記述してください）
# A: [評価] （箇条書きで記述してください）
# P: [計画] （箇条書きで記述してください）
# """









soap_prompt = f"""
The following Japanese conversation is a transcription from the original audio and may contain incompleteness and errors. Please correct it using appropriate medical terminology and use the words from the following medical terminology list as needed. If any words in the list have similar meanings, prioritize using those. Additionally, in the summary, omit subjects (e.g., doctor, patient, he, etc.) (shogu) and write concisely according to the context.

Medical terminology list: {', '.join(medical_words)}

Please provide the summary in the following structure. Write everything in Japanese.

{conversation_text}

Summary structure:
S: [Subjective information] (1 line is sufficient) (remove the subject like “patient” or “doctor,” and do not use pronouns like “he” or “she”; use concise expressions as appropriate to the context.)  (Example: 以前の病院でバセドー病と診断され、紹介状を持って新たな医療機関を受診した。)
O: [Objective information] (Write in bullet points)
A: [Assessment] (Write in bullet points)
P: [Plan] (Write in bullet points)
"""






# Step 3: Use ChatCompletion for GPT-4
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": soap_prompt}],
    max_tokens=400,
    temperature=0.5
)

# Extracting the SOAP summary
soap_summary = response['choices'][0]['message']['content']

# Save both the transcription and SOAP summary to a .txt file
with open("summary/soap_summary.txt", "w", encoding="utf-8") as output_file:
    output_file.write("Transcription:\n")
    output_file.write(conversation_text + "\n\n")
    output_file.write("SOAP Summary:\n")
    output_file.write(soap_summary)

print("Transcription and SOAP summary saved to soap_summary.txt")




# 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