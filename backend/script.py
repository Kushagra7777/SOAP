import openai
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_soap_summary(transcript_path: str, summary_types: list[str], summary_output_path: str = "summary/soap_summary.txt") -> str:
    file_path = Path(transcript_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Transcript file not found at {transcript_path}")

    with file_path.open("r", encoding="utf-8") as file:
        content = file.read()

    conversation_text = content.replace('\n', '')

    medical_words = ['バセドー病', 'クラックル音', 'チラージン']

    summary_instructions = []
    if "SOAP" in summary_types:
        summary_instructions.append("""
S: (主観的情報。患者の訴えや感情表現を含めて、2〜3文で簡潔かつ具体的に記述してください)  
O: （客観的情報。箇条書き）  
A: （評価・診断。箇条書き）  
P: （治療計画。箇条書き）
""")
    if "SAMPLE" in summary_types:
        summary_instructions.append("""
SAMPLE:  
- S(主訴・症状):  
- A(アレルギー）:  
- M(内服薬）:  
- P(既往歴）:  
- L(最終食事）:  
- E(発症前後の出来事）:
""")
    if "OPQRST" in summary_types:
        summary_instructions.append("""
OPQRST:  
- O(発症様式）:  
- P(増悪/寛解因子）:  
- Q(痛みの性質・程度）:  
- R(部位・放散）:  
- S(随伴症状）:  
- T(時間経過）:
""")

    prompt_suffix = "\n\n".join(summary_instructions)

    final_prompt = f"""
以下は、元の音声から書き起こされた日本語の会話文です。不完全な表現や誤記が含まれる可能性があります。以下の指示に従い、医療専門用語を用いて内容を正しく解釈・修正し、要約を日本語で作成してください。出力は診療記録に記載できるような簡潔で専門的な文体で記述してください。

【医療用語リスト】  
{', '.join(medical_words)}

---

【会話文】  
{conversation_text}

---

{prompt_suffix}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": final_prompt}],
        max_tokens=2000,
        temperature=0.5,
    )

    soap_summary = response.choices[0].message.content

    Path(summary_output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(summary_output_path, "w", encoding="utf-8") as output_file:
        output_file.write("Transcription:\n")
        output_file.write(conversation_text + "\n\n")
        output_file.write("Generated Summary:\n")
        output_file.write(soap_summary)

    print(f"Summary saved to {summary_output_path}")
    return soap_summary
