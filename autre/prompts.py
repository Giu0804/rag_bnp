FEW_SHOT_USER = """### CONTEXT:
ID: 1 | Paragraph: Company ABC is located in Paris and has 215 employees.
ID: 2 | Paragraph: Company ABC was founded in 2010 by Jean Dupont.
ID: 3 | Paragraph: Marie Courtois studied Law before joining Company ABC.
ID: 4 | Paragraph: Jean Dupont appointed Marie Courtois as CEO in 2018.

### QUESTION:
Who is the CEO of Company ABC?"""

FEW_SHOT_OUTPUT = """<chunks_id>2, 4</chunks_id>
<answer>Marie Courtois is the CEO of Company X.</answer>"""


messages = [
        # 1. Vos règles du jeu
        {"role": "system", "content": system_prompt},
        
        # 2. L'EXEMPLE FEW-SHOT (Input / Output)
        {"role": "user", "content": FEW_SHOT_INPUT},
        {"role": "assistant", "content": FEW_SHOT_OUTPUT},
        
        # 3. La vraie question du dataset
        {"role": "user", "content": user_prompt}
    ]