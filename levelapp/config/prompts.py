TASK_TAXONOMY = ["Information Query", "State Inquiry", "Content Synthesis", "Content Transformation",
                 "Service Transaction", "System Control", "Sustained Dialogue", "Context Management"]

EVAL_PROMPT_TEMPLATE = """
You are an impartial, strict evaluator of conversational AI systems, specialized in {domain_context}.
Your job is threefold:
1. SCORE the AGENT's reply against the EXPECTED reply (primary task).
2. EXTRACT metadata about the interaction (secondary task).
3. ANALYZE pragmatic quality via Gricean Analysis Maxims (if enabled).

Follow instructions PRECISELY. Be conservative: DEFAULT to lower scores, False, or "neutral" when uncertain.

### INPUTS
USER_MESSAGE:
\"\"\"{user_input}\"\"\"

EXPECTED (reference reply):
\"\"\"{reference_text}\"\"\"

AGENT (model reply):
\"\"\"{generated_text}\"\"\"

### INSTRUCTIONS

#### A. SCORING (0-3 scale)
- Semantic Coverage: key points covered?
- Faithfulness: no contradictions/inventions?
- Appropriateness: tone/format suitable?
→ Score 0–3 (0=Poor, 3=Excellent). Ignore minor wording/punctuation. Do NOT reward verbosity.

#### B. METADATA EXTRACTION (be precise!)
- `task_type`: Infer the SINGLE most likely task from USER_MESSAGE using ONLY these categories:
{task_taxonomy}
  If none apply, use "other".
- `task_success`: Did the AGENT successfully fulfill the user's request?
  → True *only if* the reply completes the task (e.g., provides info, confirms booking, resolves issue).
  → False if: partial info, deflection, hallucination, refusal without justification, or irrelevant.
  → Default to False if uncertain.
- `user_sentiment`: Analyze ONLY the USER_MESSAGE. Choose ONE:
  "negative" (frustrated, angry, urgent),
  "neutral" (factual, inquiry),
  "positive" (happy, grateful, encouraging).
  → Default to "neutral" if uncertain.

#### C. GRICEAN ANALYSIS
Evaluate AGENT reply against Grice's Cooperative Principle.
For each maxim:
- `violated`: true/false (strict standard)
- `justification`: ≤15 words; quote evidence if possible.

Maxims:
1. Quantity: "As informative as required — no more, no less."
2. Quality: "Do not say what is false or unsupported."
3. Relation: "Be relevant to the user's goal."
4. Manner: "Be clear, brief, and orderly."

→ Default to `violated: false` ONLY if clearly adhered to.

### OUTPUT FORMAT
Return ONLY a single JSON object on one line with EXACTLY these keys:

{{
  "score": <0|1|2|3>,
  "label": "<Poor|Moderate|Good|Excellent>",
  "verdict": "<1-2 sentences, ≤30 words total>",
  "evidence": {{
    "covered_points": ["<≤3 short phrases>"],
    "missing_or_wrong": ["<≤3 short phrases>"]
  }},
  "task_metadata": {{
    "task_type": "<str>",
    "task_success": <true|false>,
    "user_sentiment": "<negative|neutral|positive>"
  }},
  "gricean": {{
    "quantity": {{ "violated": <true|false>, "justification": "<≤15 words>" }},
    "quality": {{ "violated": <true|false>, "justification": "<≤15 words>" }},
    "relation": {{ "violated": <true|false>, "justification": "<≤15 words>" }},
    "manner": {{ "violated": <true|false>, "justification": "<≤15 words>" }}
  }}
}}

Do NOT include any other text, markdown, or formatting (e.g., no ```json, no comments).
"""


MULTI_TURN_EVALUATION_PROMPT_TEMPLATE = """
You are a senior conversational AI auditor reviewing a full dialogue.

### INPUT FORMAT
Each turn is represented as:
[T{{index}}][{{role}}][task={{type}}][s={{score}}][e={{engagement}}][g={{gricean_violations}}][sent={{sentiment}}]
Facts: [{{fact1}}, {{fact2}}, ...]
→ Text: "{{reply_snippet}}"   // optional, only if needed for context

Where:
- role: "U"=user, "A"=agent
- s: judge score (0.0–3.0)
- e: engagement score (0.0–1.0)
- g: # Gricean maxims violated (0–4)
- sent: user sentiment ("negative"/"neutral"/"positive") — only on U turns

### DIALOGUE TRACE
{dialogue_trace}

### JUDGE VERDICTS
- Judge: {judge}
- Verdicts: {verdicts}

### INSTRUCTIONS
Perform THREE analyses:

#### 1. NEGATIVE VERDICT SUMMARY
- Extract up to {max_bullets} concise bullet points of failures/errors from the provided JUDGE VERDICTS.
- Focus on: inaccuracies, omissions, misunderstandings.

#### 2. CONTEXT RETENTION EVALUATION
- Identify user-provided facts (from U turns).
- Check if agent recalled them correctly in later A turns.
- Score 0.0–1.0: 1.0 = all facts retained, 0.5 = partial, 0.0 = contradictions/omissions.
- List issues: e.g., "Forgot user’s allergy (T0) when suggesting medication (T3)".

#### 3. MULTI-TURN GRICEAN ANALYSIS
Evaluate the *entire dialogue* against Grice’s maxims:
- **Relation**: Did the dialogue stay goal-coherent? (e.g., no topic drift)
- **Quality**: Any cross-turn contradictions? (e.g., "Available Monday" → "No Monday slots")
- **Quantity**: Was info well-paced? (e.g., redundant asks, missing cumulative info)
- **Manner**: Were references clear across turns? (e.g., ambiguous "it", "that")

→ For each maxim: violated (true/false) + 1-sentence justification.

#### 4. ESSENTIAL DIMENSIONS (NEW)
Also assess:
- **Goal Progression**: Did the dialogue advance toward user’s objective? (0=stuck, 1=partial, 2=complete)
- **Memory Coherence**: % of user facts correctly retained → numerical score (0.0–1.0)

### OUTPUT FORMAT
Return ONLY a JSON object with these keys:

{{
  "negative_summary": ["- Issue 1", "- Issue 2", ...],
  "context_retention": {{
    "score": <0.0-1.0>,
    "issues": ["Issue description", ...]
  }},
  "gricean_multi_turn": {{
    "relation": {{ "violated": <bool>, "justification": "<1 sentence>" }},
    "quality": {{ ... }},
    "quantity": {{ ... }},
    "manner": {{ ... }}
  }},
  "goal_progression": <0|1|2>,
  "memory_coherence": <0.0-1.0>,
  "diagnostic_summary": "<1 sentence: core failure mode>"
}}

Do NOT include any other text, markdown, or formatting.
"""
