EVAL_PROMPT_TEMPLATE = """
You are an impartial evaluator for a conversational system.
Compare the AGENT's reply to the EXPECTED reply for the SAME user message.

Consider only:
1) Semantic Coverage — does the AGENT cover the key points in EXPECTED?
2) Faithfulness — no contradictions or invented details relative to EXPECTED.
3) Appropriateness — tone/format suitable for the user message.
Ignore minor wording/punctuation differences. Do NOT reward verbosity.

Scale (integer):
0 = Poor (misses key points or contradicts)
1 = Moderate (captures some ideas, noticeable gaps)
2 = Good (mostly matches, minor omissions/differences)
3 = Excellent (semantically equivalent; no meaningful differences)

USER_MESSAGE:
\"\"\"{user_input}\"\"\"

EXPECTED (reference reply):
\"\"\"{reference_text}\"\"\"

AGENT (model reply):
\"\"\"{generated_text}\"\"\"

Return ONLY a single JSON object on one line with exactly these keys:
- "score": <0|1|2|3>,
- "label": "<Poor|Moderate|Good|Excellent>",
- "justification": "<1-2 concise sentences>",
- "evidence":
    - "covered_points": ["<short phrase>", "..."],   // <=3 items
    - "missing_or_wrong": ["<short phrase>", "..."]  // <=3 items
    
Do NOT include any additional text, explanations, or formatting (e.g., "JSON object:", ```json or ```, or markdown).
"""
