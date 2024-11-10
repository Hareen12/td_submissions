from openai import OpenAI
import base64
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from submission_1 import decode
#encrypted keys
client = OpenAI(api_key = base64.b64decode(decode.temp).decode("utf-8") )
def LLM_model(words):
    # words = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape', 'honeydew', 'kiwi', 'lemon', 'mango', 'nectarine', 'orange', 'papaya', 'quince', 'raspberry']
    s="""
    Find groups of four items that share something in common.
    Category Examples

    FISH: Bass, Flounder, Salmon, Trout
    FIRE ___:
    Ant, Drill, Island, Opal
    Categories will always be more specific than
    "5-LETTER-WORDS," "NAMES" or "VERBS."

    Each puzzle has exactly one solution. Every item fits in
    exactly one category.

    Watch out for words that seem to belong to multiple categories!

    Order your answers in terms of your confidence level, high
    confidence first.

    Here are the items: 
        """
    prompt =s 

    prompt+=str(words)
    s1 = """

    Return your guess as ONLY JSON like this:

    {"groups":
        [
            ["item1a", "item2a", "item3a", "item4a"],["item2a", "item2b", "item3b", "item4b"],
        ]}

    No other text.
    """
    prompt+=s1
    # print(prompt)
    completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="gpt-4o-mini",
        # model ="o1-preview",
        max_tokens = 100
    )

    m=completion.choices[0].message
    # print(m)

    response = m.content
    response = eval(response)
    l = response["groups"]
    # print("l is ",l)
    # print(type(l))
    return l

# words = "['RADICAL', 'LICK', 'EXPONENT', 'SHRED', 'GNARLY', 'ROOT', 'OUNCE','TWISTED', 'THRONE', 'TRACE', 'BATH', 'BENT', 'REST', 'POWDER', 'POWER','WARPED']"
# words = "['LICK', 'SHRED', 'GNARLY', 'OUNCE','TWISTED', 'THRONE', 'TRACE', 'BATH', 'BENT', 'REST', 'POWDER','WARPED']"

# print(LLM_model(words))