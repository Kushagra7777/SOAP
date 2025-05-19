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
頭痛, 胸痛, 発熱, 咳, 倦怠感, 腹痛, めまい, 息切れ, 吐き気, 喉の痛み  

＜アレルギー＞  
ペニシリンアレルギー, ピーナッツアレルギー, ラテックスアレルギー, 甲殻類アレルギー, 花粉症, ダニアレルギー, 蜂毒アレルギー, NSAIDsアレルギー, スルファ薬アレルギー, 卵アレルギー, アスピリンアレルギー, 魚アレルギー  

＜内服薬＞  
アセトアミノフェン, 血圧の薬, 抗生物質, インスリン, ビタミンD, アスピリン, メトホルミン, リシノプリル, アトルバスタチン, アムロジピン, オメプラゾール, ワルファリン, レボチロキシン, アルブテロール, 葛根湯（かっこんとう）, 小青竜湯（しょうせいりゅうとう）  

＜既往歴＞  
糖尿病, 高血圧, 心臓病, 胃潰瘍, 喘息  

＜食事＞  
朝食を食べた, 昼食なし, 夕食前の薬, 飲み物を摂った, 食べ物の摂取なし  

＜発症時の出来事＞  
事故後の痛み, 運動中の怪我, 夜間の咳, 食事後の吐き気, 寒気が続いている  

＜発症様式＞  
急に始まった, 徐々に始まった, 昨日から, 先週から, 3日間, 断続的, 持続的, 初めて, 繰り返す, 急性, 今朝から, 1時間前に, 数日前から  

＜増悪/寛解因子＞  
動かすと悪化, 食事で悪化, 休むと軽快, 薬で良くなる, ストレスで悪化, 寒さで悪化, 温めると楽になる, 深呼吸で悪化, 横になると軽快, 何をしても良くならない, 何かを食べた後に悪化する, 座っていると痛みが軽くなる, 歩くと痛みが強くなる  

＜性質と痛みの程度＞  
鋭い痛み, 鈍い痛み, 焼けるような痛み, ズキズキする痛み, けいれん性の痛み, 圧迫感, 刺すような痛み, しびれる感じ, うずくような痛み, 締め付け感, 激しい痛み, 10段階中8の痛み  

＜部位と放散＞  
胸部, 腕に放散, 背中に放散, 右下腹部, 胸全体, 目の奥, 顎の痛み, 脚に広がる, 上腹部, 右肩, 肩から腕に放散, 下腹部  

＜随伴症状＞  
発熱, 吐き気, 嘔吐, 悪寒, めまい, 息切れ, 倦怠感, 発汗, 頭痛, しびれ, 吐き気を伴う, 立ちくらみ, 動悸を感じる, 呼吸困難  

＜時間経過＞  
今朝から, 1時間続いている, 数分間続いている, 昨日の夜から, 1週間前に始まった, 間欠的に発生, 出たり消えたり, 突然現れてすぐ消える, 慢性的, 運動後に始まる, 夜に悪化, 食後に出現, 毎日, 週に1回

---

【会話文】  
{conversation_text}

---

【出力フォーマット】  
以下の形式に従って、会話内容の要約を記述してください（すべて日本語）。

S: (主観的情報。患者の訴えや感情表現を含めて、2〜3文で簡潔かつ具体的に記述してください)  
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
        model="gpt-4o",
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


# S: (主観的情報。1行で簡潔に)