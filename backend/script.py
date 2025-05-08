import openai
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_soap_summary(transcript_path: str, summary_output_path: str = "summary/soap_summary.txt") -> str:
    file_path = Path(transcript_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Transcript file not found at {transcript_path}")

    with file_path.open("r", encoding="utf-8") as file:
        content = file.read()

    conversation_text = content.replace('\n', '')

    medical_words = ['バセドー病', 'クラックル音', 'チラージン']

    soap_prompt = f"""
以下は、元の音声から書き起こされた日本語の会話文です。不完全な表現や誤記が含まれる可能性があります。以下の指示に従い、医療専門用語を用いて内容を正しく解釈・修正し、SOAP形式およびSAMPLE + OPQRSTに基づいた要約を日本語で作成してください。出力は診療記録に記載できるような簡潔で専門的な文体で記述してください。

### 指示:
1. 以下の医療用語リストに含まれる単語は、意味が近い場合に優先的に使用してください。
2. 会話の内容から明らかに読み取れる場合は、推論に基づいて情報を補完しても構いません（例：明確な痛みの訴えがある場合、それに基づく推論を記述してよい）。
3. 要約では主語（患者、彼、彼女など）を省略し、文脈から明らかな情報は簡潔かつ断定的に述べてください。
4. SAMPLEおよびOPQRSTフレームワークに沿って情報を分類し、適切に記載してください。
5. 出力は以下の形式に厳密に従ってください。すべて日本語で記述してください。

---

【医療用語リスト】  
{', '.join(medical_words)}

---

【SAMPLEフレームワーク】
S(主訴・症状）  
A(アレルギーの有無）  
M(内服薬）  
P(既往歴）  
L(最終食事摂取）  
E(発症時の出来事）

【OPQRSTフレームワーク】
O(発症様式）  
P(増悪・寛解因子）  
Q(性質と痛みの程度）  
R(部位と放散）  
S(随伴症状）  
T(時間経過）

---

【参考語彙】（使える場合は優先的に使用してください）

＜症状＞
発熱, 頭痛, 咳, 息切れ, 倦怠感, 吐き気, 嘔吐, 下痢, 腹痛, 胸痛, 動悸, めまい  
＜アレルギー＞
ペニシリンアレルギー, ラテックスアレルギー, 花粉症, アスピリンアレルギー, 魚アレルギー  
＜内服薬＞
アセトアミノフェン, 血圧の薬, 抗生物質, インスリン, ビタミンD  
＜既往歴＞
糖尿病, 高血圧, 心臓病, 胃潰瘍, 喘息  
＜食事＞
朝食を食べた, 昼食なし, 夕食前の薬, 飲み物を摂った, 食べ物の摂取なし  
＜発症時の出来事＞
事故後の痛み, 運動中の怪我, 夜間の咳, 食事後の吐き気, 寒気が続いている  
＜発症様式＞
今朝から, 1時間前に, 突然, 昨日から, 数日前から  
＜増悪/寛解因子＞
何かを食べた後に悪化する, 休むと楽になる, 座っていると痛みが軽くなる, 歩くと痛みが強くなる, 薬を飲んで楽になる  
＜性質と痛みの程度＞
鋭い痛み, 鈍い痛み, 締め付けられる感じ, 激しい痛み, 圧迫感, 10段階中8の痛み  
＜部位と放散＞
胸部, 背中に広がる, 右側の腹部, 足に広がる, 肩から腕に放散, 下腹部  
＜随伴症状＞
吐き気を伴う, 立ちくらみ, 動悸を感じる, 呼吸困難  
＜時間経過＞
1時間続いている, 数分間続いている, 昨日の夜から, 1週間前に始まった, 間欠的に発生

---

【会話文】
{conversation_text}

---

【出力フォーマット】
以下の形式に従って、会話内容の要約を記述してください（すべて日本語）。

S: (主観的情報。1行で簡潔に)
O: （客観的情報。箇条書き）  
A: （評価・診断。箇条書き）  
P: （治療計画。箇条書き）

SAMPLE:
- S(主訴・症状): 
- A(アレルギー）: 
- M(内服薬）: 
- P(既往歴）: 
- L(最終食事）: 
- E(発症前後の出来事）: 

OPQRST:
- O(発症様式）: 
- P(増悪/寛解因子）: 
- Q(痛みの性質・程度）: 
- R(部位・放散）: 
- S(随伴症状）: 
- T(時間経過）:
"""

    # Use the new API format
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": soap_prompt}],
        max_tokens=2000,
        temperature=0.5,
    )

    soap_summary = response.choices[0].message.content

    Path(summary_output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(summary_output_path, "w", encoding="utf-8") as output_file:
        output_file.write("Transcription:\n")
        output_file.write(conversation_text + "\n\n")
        output_file.write("SOAP Summary:\n")
        output_file.write(soap_summary)

    print(f"SOAP summary saved to {summary_output_path}")
    return soap_summary