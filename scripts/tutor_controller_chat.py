from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional, Tuple
import re
import json
import ollama

RetrieveFn = Callable[[str, int], List[Dict[str, Any]]]  # [{"id":..., "score":..., "text":...}, ...]


# ============================================================
# Curriculum (unchanged)
# ============================================================
CURRICULUM = [
    {
        "id": "1.1",
        "unit_num": 1,
        "unit_title": "Foundations of Computing",
        "chapter": "What is a Computer?",
        "outline": [
            "Definition of a computer",
            "Origin of the word 'compute'",
            "Data and processing concept",
            "Computer as input → process → output system",
        ],
        "learning_objective": "Student can explain what a computer is in their own words.",
    },
    {
        "id": "1.2",
        "unit_num": 1,
        "unit_title": "Foundations of Computing",
        "chapter": "Characteristics of a Computer",
        "outline": [
            "Speed",
            "Accuracy",
            "Automation",
            "Diligence",
            "Versatility",
            "Storage capacity",
        ],
        "learning_objective": "Student can list and explain at least three characteristics with examples.",
    },
    {
        "id": "1.3",
        "unit_num": 1,
        "unit_title": "Foundations of Computing",
        "chapter": "Types of Computers (By Data Processed)",
        "outline": [
            "Analog computers",
            "Digital computers",
            "Hybrid computers",
            "Differences and use cases",
        ],
        "learning_objective": "Student can compare analog and digital systems conceptually.",
    },
    {
        "id": "2.1",
        "unit_num": 2,
        "unit_title": "Input and Output Systems",
        "chapter": "Input Devices",
        "outline": [
            "Keyboard",
            "Mouse",
            "Trackball",
            "Scanner",
            "Microphone",
            "Other pointing devices",
        ],
        "learning_objective": "Student can classify a device as input and explain its function.",
    },
    {
        "id": "2.2",
        "unit_num": 2,
        "unit_title": "Input and Output Systems",
        "chapter": "Pointing Devices",
        "outline": [
            "Mouse",
            "Trackball",
            "Touchpad",
            "Light pen",
            "Stylus",
            "Joystick",
        ],
        "learning_objective": "Student can distinguish pointing devices from general input devices.",
    },
    {
        "id": "2.3",
        "unit_num": 2,
        "unit_title": "Input and Output Systems",
        "chapter": "Output Devices",
        "outline": [
            "Monitor",
            "Printer",
            "Speaker",
            "Plotter",
        ],
        "learning_objective": "Student understands output as processed information.",
    },
    {
        "id": "3.1",
        "unit_num": 3,
        "unit_title": "Memory and Storage",
        "chapter": "Primary Memory",
        "outline": [
            "RAM",
            "ROM",
            "Volatile vs non-volatile",
        ],
        "learning_objective": "Student can differentiate memory types and explain purpose.",
    },
    {
        "id": "3.2",
        "unit_num": 3,
        "unit_title": "Memory and Storage",
        "chapter": "Secondary Storage",
        "outline": [
            "Hard disk",
            "SSD",
            "USB",
            "CD/DVD",
        ],
        "learning_objective": "Student can differentiate memory types and explain purpose.",
    },
    {
        "id": "4.1",
        "unit_num": 4,
        "unit_title": "Operating Systems",
        "chapter": "What is an Operating System?",
        "outline": [
            "Definition",
            "Role as intermediary",
            "Examples",
        ],
        "learning_objective": "Student can explain why a computer needs an OS.",
    },
    {
        "id": "4.2",
        "unit_num": 4,
        "unit_title": "Operating Systems",
        "chapter": "Functions of OS",
        "outline": [
            "Process management",
            "Memory management",
            "File systems",
            "Device management",
        ],
        "learning_objective": "Student can explain why a computer needs an OS.",
    },
    {
        "id": "5.1",
        "unit_num": 5,
        "unit_title": "Computer Languages",
        "chapter": "Machine Language",
        "outline": [
            "Binary",
            "Direct hardware interaction",
        ],
        "learning_objective": "Student understands abstraction levels in programming languages.",
    },
    {
        "id": "5.2",
        "unit_num": 5,
        "unit_title": "Computer Languages",
        "chapter": "Assembly Language",
        "outline": [
            "Mnemonic instructions",
            "Closer to hardware than high-level languages",
        ],
        "learning_objective": "Student understands abstraction levels in programming languages.",
    },
    {
        "id": "5.3",
        "unit_num": 5,
        "unit_title": "Computer Languages",
        "chapter": "High-Level Languages",
        "outline": [
            "Human-friendly syntax",
            "Examples (e.g., Python, Java, C++)",
            "Abstraction vs machine language",
        ],
        "learning_objective": "Student understands abstraction levels in programming languages.",
    },
]


def curriculum_ids() -> List[str]:
    return [c["id"] for c in CURRICULUM]


def get_item(lesson_id: str) -> Dict[str, Any]:
    for c in CURRICULUM:
        if c["id"] == lesson_id:
            return c
    return CURRICULUM[0]


def next_lesson_id(current_id: str) -> str:
    ids = curriculum_ids()
    if current_id not in ids:
        return ids[0]
    i = ids.index(current_id)
    return ids[min(i + 1, len(ids) - 1)]


def first_lesson_of_unit(unit_num: int) -> str:
    for c in CURRICULUM:
        if c["unit_num"] == unit_num:
            return c["id"]
    return CURRICULUM[0]["id"]


def curriculum_text_for_welcome() -> str:
    """Units-only list for onboarding."""
    by_unit: Dict[int, List[Dict[str, Any]]] = {}
    for c in CURRICULUM:
        by_unit.setdefault(c["unit_num"], []).append(c)
    lines: List[str] = []
    for u in sorted(by_unit.keys()):
        lines.append(f"Unit {u}: {by_unit[u][0]['unit_title']}")
    return "\n".join(lines).strip()


def unit_title_map() -> Dict[str, int]:
    """Map normalized unit titles to unit numbers for tolerant onboarding parsing."""
    def _n(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())
    m: Dict[str, int] = {}
    for c in CURRICULUM:
        title = (c.get("unit_title") or "").strip()
        if not title:
            continue
        m[_n(title)] = int(c["unit_num"])
    return m


def parse_lesson_or_unit_command(text: str) -> Tuple[Optional[str], Optional[int], bool]:
    t = text.lower().strip()
    start = any(p in t for p in [
        "start from the beginning", "start from the start",
        "begin from the start", "from the start", "from scratch",
    ])
    m = re.search(r"\blesson\s*(\d\.\d)\b", t)
    if m and m.group(1) in set(curriculum_ids()):
        return m.group(1), None, start
    m2 = re.search(r"\b(\d\.\d)\b", t)
    if m2 and m2.group(1) in set(curriculum_ids()):
        return m2.group(1), None, start
    mu = re.search(r"\bunit\s*(\d)\b", t)
    if mu:
        return None, int(mu.group(1)), start

    # Allow bare unit numbers: "1" / "2" / "5"
    if re.fullmatch(r"\d", t) and 1 <= int(t) <= 5:
        return None, int(t), start

    # Allow unit title selection by name (e.g., "Foundations of Computing")
    utm = unit_title_map()
    tn = _norm(text)
    for title_norm, unit_num in utm.items():
        # substring match so "let's do foundations of computing" works
        if title_norm and title_norm in tn:
            return None, unit_num, start
    return None, None, start


def build_optional_notes(retrieved: List[Dict[str, Any]], max_chunks: int = 2, max_chars: int = 900) -> Tuple[str, float]:
    if not retrieved:
        return "Optional course material: (none found)", -1.0
    best = max(float(r.get("score", -1.0)) for r in retrieved)
    lines = ["Optional course material (use if helpful):"]
    for r in retrieved[:max_chunks]:
        txt = (r.get("text") or "").strip()
        if len(txt) > max_chars:
            txt = txt[:max_chars].rstrip() + "..."
        lines.append(txt)
    return "\n".join(lines), best



def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())



ASSESSOR_PROMPT_FULL = """
You are evaluating a tutoring interaction.

Assess conversationally (not like an exam marker):
- Accept casual phrasing if the idea is correct.
- If the learner asks a question (ends with ? or is clearly a question) or expresses interest (e.g., “wow”), treat it as engagement: last_turn_label="ACK" and momentum_delta=0.
- Only mark INCORRECT when the learner makes a clearly wrong claim relative to the tutor's last question.

You must do TWO things:

1) Decide whether the learner has met the chapter learning objective based on their replies in the full chapter transcript.
   - Only set advance=true if the transcript contains evidence the learning objective is met.
   - In chapter_reason, reference at least ONE concrete element the learner did (or did not) demonstrate.
2) Evaluate the learner's MOST RECENT reply to the tutor's most recent question.
   - Provide a label and the kind of support needed next.
   - Provide a momentum_delta in [-3, +3] reflecting answer quality and engagement
   - Do not give negative points for correct or partially correct answers
   - Here is how to rate momentum:
     * +3 = strong correct, clear explanation in their own words
     * +2 = correct
     * +1 = partially correct / on track
     *  0 = acknowledgement / engagement / learner question (e.g., "ok", "thanks", "wow", or asking a relevant counter-question)
     * -1 = incorrect
     * -2 = confused / very weak / "I don't know"
     * -3 = off-topic derailment or refusal to engage
Teaching style mode for this turn: {teaching_style}
- EASY: be lenient. Prefer PARTIAL over INCORRECT for vague-but-related answers. Use smaller penalties; reserve INCORRECT for clearly wrong claims.
- NORMAL: balanced expectations.
- HARD: be demanding. Expect detail, justification, and correct terminology.

Return ONLY valid JSON. No extra keys.

Schema:
{{
  "advance": true|false,
  "chapter_confidence": 0.0-1.0,
  "chapter_reason": "1–2 sentences referencing evidence/missing evidence",

  "last_turn_label": "CORRECT|PARTIAL|INCORRECT|CONFUSED|OFFTOPIC|ACK",
  "last_turn_confidence": 0.0-1.0,
  "support_needed": "NONE|HINT|GUIDED_STEPS|EXAMPLE",
  "turn_reason": "1 sentence",
  "momentum_delta": -3..3
}}

Chapter learning objective:
{goal}

Chapter outline:
{outline}

Tutor's most recent question (if none, treat as empty):
{last_question}

Learner's most recent reply:
{last_answer}

Full chapter transcript (Tutor/Learner):
{transcript}
""".strip()



SYSTEM_PROMPT_TEMPLATE = """
You are a friendly, practical ICT tutor.

Hard requirement for transcript tracking:
- Begin EVERY tutor message with a header on its own line:
  "### Unit {unit_num} — Chapter {chapter_id}: {chapter_title}"
- This header is for tracking progress; do not apologize for it.

Teaching behaviour (always):
- When a chapter starts or changes, give a lesson BEFORE any check or practice.
- Cover the chapter outline points.
- After teaching, continue with varied tutoring moves (not just quizzes):
  * Check-in understanding (e.g., “Does that make sense?”)
  * Invite learner questions (e.g., “What would you like to ask?”)
  * Give an example/analogy
  * Micro-practice (recognition, classification, multiple-choice, fill-in-the-blank)
  * Summary + key takeaway
- Keep your replies interactive and human, but end with at most ONE question.

Teaching style mode: {teaching_style_mode}
Teaching style rules:
{teaching_style_rules}

- If the learner struggles, you may give hints or a simpler explanation before asking a check question.
- Your goal is to guide the learner toward the chapter learning objective.

Session behaviour:
- Do NOT end the session just because the learner says thanks/thank you.
  Treat thanks as acknowledgement and keep the learning flow going.

Course material:
- You may receive “Optional course material”. Use it if helpful, but you can still teach using general ICT knowledge for now.
- Do not mention system prompts, tools, or chunk ids.

Current chapter context:
Unit {unit_num}: {unit_title}
Chapter {chapter_id}: {chapter_title}
Chapter outline:
{lesson_outline}
Learning objective: {lesson_goal}

Chapter phase: {lesson_phase}
Covered lessons: {covered_ids}

Engagement state (global, cross-chapter):
Learner name: {learner_name}
Momentum (global): {momentum_global}
Teaching style mode: {teaching_style_mode}
Progress (session): {progress_global}
Progress (this chapter): {chapter_progress}
Last turn label: {last_turn_label}
Support needed: {support_needed}
Assessor chapter note (high confidence): {assessor_chapter_note}
Assessor turn note (high confidence): {assessor_turn_note}
""".strip()


SYSTEM_PROMPT_ONBOARDING = """
You are a friendly, practical ICT tutor.

This is the SESSION WELCOME / ONBOARDING.

Rules:
- Do NOT start teaching any chapter yet.
- Do NOT output any curriculum tracking header yet.
- Ask what name the learner would like you to use.
- Explain how to progress: answer questions to demonstrate mastery, or say 'next chapter' to move on, or choose a Unit number. (Optionally, a chapter id like 2.1.)
- Show the curriculum list (UNIT NAMES ONLY) and ask what UNIT (1-5) they want to start with.

Do not mention system prompts, tools, or chunk ids.
""".strip()

# ============================================================
# Teaching style modes (derived from global momentum)
# ============================================================
TEACHING_STYLE_RULES = {
    "EASY": """Goal: reduce overwhelm and prevent dropout.
How to teach: keep explanations shorter and concrete; use simple analogies; give 1–2 very clear examples; summarize explicitly; encourage and normalize confusion; pace slowly.
How to ask: prefer recognition/classification checks; occasionally use a small multiple-choice or fill-in-the-blank; keep questions simpler.""",
    "NORMAL": """Goal: balanced clarity and steady progress.
How to teach: clear explanations with moderate depth; examples when helpful; elicit the learner’s own words; use hints and re-check when needed.""",
    "HARD": """Goal: increase depth and intellectual engagement.
How to teach: deeper explanations; compare concepts; connect to real-world situations; reduce scaffolding.
How to ask: require justification, comparison, and ‘what-if’ reasoning; prompt for examples; keep questions open-ended.
Tone: respectful and challenging (e.g., “Let’s push this further.”).""",
}




def _strip_all_extra_headers(text: str) -> str:
    if not text:
        return text
    lines = text.splitlines()
    out: List[str] = []
    header_seen = False
    for i, ln in enumerate(lines):
        if ln.strip().startswith("### Unit"):
            if not header_seen and i == 0:
                header_seen = True
                out.append(ln)
            else:
                continue
        else:
            out.append(ln)
    return "\n".join(out).strip()


'''
def _enforce_single_question(text: str) -> str:
    """Enforce at most ONE question per tutor message by keeping the LAST '?' in the body."""
    if not text or '?' not in text:
        return text

    lines = text.splitlines()
    header = ''
    body_lines = lines
    if lines and lines[0].strip().startswith('### Unit'):
        header = lines[0]
        body_lines = lines[1:]

    body = "\n".join(body_lines)
    if '?' not in body:
        return text

    last_q = body.rfind('?')
    if last_q <= 0:
        return text

    before = body[:last_q].replace('?', '.')
    after = body[last_q:]
    body2 = (before + after).strip()

    if header:
        return (header + "\n" + body2).strip()
    return body2'''

def _extract_last_tutor_question(history: List[Dict[str, str]], max_scan: int = 10) -> str:
    scan = history[-max_scan:] if max_scan > 0 else history
    for msg in reversed(scan):
        if msg.get("role") == "assistant":
            txt = (msg.get("content") or "").strip()
            if not txt:
                continue
            lines = txt.splitlines()
            if lines and lines[0].strip().startswith("### Unit"):
                txt = "\n".join(lines[1:]).strip() or txt
            qs = re.findall(r"([^?\n]{1,220}\?)", txt)
            if qs:
                return qs[-1].strip()
    return ""


@dataclass
class TutorSession:
    retrieve_fn: RetrieveFn
    model: str = "llama3.2:3b-instruct-q4_K_M"
    include_welcome: bool = True
    debug: bool = True

    use_assessor: bool = True
    assessor_model: Optional[str] = None
    advance_conf_threshold: float = 0.66
    assessor_debug: bool = True

    history: List[Dict[str, str]] = field(default_factory=list)

    session_phase: str = "ONBOARDING"
    learner_name: Optional[str] = None

    current_lesson_id: str = "1.1"
    covered_lessons: List[str] = field(default_factory=list)

    lesson_phase: str = "INTRO"
    turn_count: int = 0
    short_reply_streak: int = 0
    last_best_score: float = -1.0

    chapter_history: List[Dict[str, str]] = field(default_factory=list)

    chapter_turns: int = 0  # learner turns within current chapter
    last_chapter_reason: str = ""
    last_chapter_confidence: float = 0.0

    # Turn-level assessor signal (used to nudge the tutor; only trusted when confidence is high)
    last_turn_reason: str = ""
    last_turn_confidence: float = 0.0

    momentum_global: int = 0

    # Progress framing (self-referenced progress / milestones)
    progress_global: int = 0  # cumulative positive progress points across session
    chapter_progress: int = 0  # cumulative positive progress points within current chapter

    # Debug-only: last selected tutoring strategy label
    last_strategy: str = "N/A"
    last_turn_label: str = "N/A"
    support_needed: str = "N/A"

    move_cycle: int = 0  # cycles pedagogical move variety

    def _dbg(self, msg: str):
        if self.debug:
            print(msg)

    def _assess_dbg(self, msg: str):
        if self.debug and self.assessor_debug:
            print(msg)

    def _teaching_style_mode(self) -> str:
        # Derived from global momentum (cross-chapter)
        if self.momentum_global <= -2:
            return "EASY"
        if self.momentum_global >= 2:
            return "HARD"
        return "NORMAL"

    def _system_prompt(self) -> str:
        item = get_item(self.current_lesson_id)
        outline = "\n".join([f"- {x}" for x in item["outline"]])
        covered = ", ".join(self.covered_lessons[-12:]) if self.covered_lessons else "(none)"
        # Backwards compatible: older curricula used 'learning_objective'.
        learning_objective = item.get("learning_objective", item.get("learning_objective", ""))
        last_turn_conf = getattr(self, "last_turn_confidence", 0.0)
        last_turn_reason = getattr(self, "last_turn_reason", "")
        return SYSTEM_PROMPT_TEMPLATE.format(
            unit_num=item["unit_num"],
            unit_title=item["unit_title"],
            chapter_id=item["id"],
            chapter_title=item.get("chapter", item.get("lesson_title", "")),
            lesson_outline=outline,
            lesson_goal=learning_objective,
            lesson_phase=self.lesson_phase,
            covered_ids=covered,
            learner_name=self.learner_name or "(unknown)",
            momentum_global=str(self.momentum_global),
            progress_global=str(self.progress_global),
            chapter_progress=str(self.chapter_progress),
            teaching_style_mode=self._teaching_style_mode(),
            teaching_style_rules=TEACHING_STYLE_RULES.get(self._teaching_style_mode(), TEACHING_STYLE_RULES["NORMAL"]),
            last_turn_label=self.last_turn_label,
            support_needed=self.support_needed,
            assessor_chapter_note=(self.last_chapter_reason if self.last_chapter_confidence >= self.advance_conf_threshold else "(none)"),
            assessor_turn_note=(last_turn_reason if last_turn_conf >= self.advance_conf_threshold else "(none)"),
        )

    def _current_header_line(self) -> str:
        item = get_item(self.current_lesson_id)
        title = item.get("chapter", item.get("lesson_title", ""))
        return f"### Unit {item['unit_num']} — Chapter {item['id']}: {title}"

    def _enforce_header(self, tutor_text: str) -> str:
        header = self._current_header_line()
        if not tutor_text:
            return header
        lines = tutor_text.splitlines()
        if not lines:
            return header
        if lines[0].strip().startswith("### Unit"):
            lines[0] = header
            return _strip_all_extra_headers("\n".join(lines).strip())
        return _strip_all_extra_headers((header + "\n\n" + tutor_text).strip())

    def _chat(self, messages: List[Dict[str, str]], temperature: float = 0.82, num_predict: int = 750) -> str:
        resp = ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature, "num_predict": num_predict},
        )
        return (resp["message"]["content"] or "").strip()

    def _chapter_transcript_text(self, max_turns: int = 18, max_chars: int = 4200) -> str:
        turns = self.chapter_history[-max_turns:] if max_turns > 0 else self.chapter_history
        lines: List[str] = []
        for t in turns:
            role = t.get("role", "")
            content = (t.get("content") or "").strip()
            if not content:
                continue
            if role == "assistant":
                c_lines = content.splitlines()
                if c_lines and c_lines[0].strip().startswith("### Unit"):
                    content = "\n".join(c_lines[1:]).strip() or content
                lines.append("Tutor: " + content)
            elif role == "user":
                lines.append("Learner: " + content)
        transcript = "\n\n".join(lines).strip()
        if len(transcript) <= max_chars:
            return transcript
        return ("..." + transcript[-max_chars:]).strip()

    
    def _chapter_has_min_exchange(self) -> bool:
        """
        Gate chapter advancement to avoid 'instant skipping'.
        Requires at least ONE real back-and-forth inside the current chapter:
        - Tutor asked a question (a '?' in an assistant message), and
        - Learner provided a non-ACK reply after that question.
        """
        saw_question = False
        for msg in self.chapter_history:
            if msg.get("role") == "assistant":
                txt = (msg.get("content") or "")
                if "?" in txt:
                    saw_question = True
            elif msg.get("role") == "user" and saw_question:
                t = (msg.get("content") or "").strip()
                if not t:
                    continue
                if len(_norm(t)) >= 3:
                    return True
        return False

    def _enter_lesson(self, lesson_id: str):
        self.current_lesson_id = lesson_id
        if lesson_id not in self.covered_lessons:
            self.covered_lessons.append(lesson_id)
        self.lesson_phase = "INTRO"
        self.chapter_history = []
        self.chapter_turns = 0
        self.last_chapter_reason = ""
        self.last_chapter_confidence = 0.0
        self.last_turn_reason = ""
        self.last_turn_confidence = 0.0
        self.session_phase = "IN_CHAPTER"

        self.chapter_progress = 0
        self.last_strategy = "LESSON_INTRO"
        # Reset the pedagogical move cycle on chapter change so each chapter starts consistently.
        self.move_cycle = 0
    def start(self) -> str:
        self.session_phase = "ONBOARDING"
        if not self.include_welcome:
            return ""
        curriculum_list = curriculum_text_for_welcome()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_ONBOARDING},
            {"role": "user", "content": (
                "Welcome the learner to the ICT Fundamentals tutoring session.\n"
                "Explain briefly what you can do: teach lessons, explain concepts, give hints, give examples, ask practice questions, and give feedback.\n"
                "Explain how to progress: answer questions to demonstrate mastery, or explicitly say 'next chapter' to move on, or choose a Unit number.\n"
                "Then show the curriculum below exactly as a list of UNIT NAMES ONLY, and ask which UNIT (1-5) they want to study first.\n"
                "Don't forget to ask what name they'd like to use during the session.\n\n"
                "Curriculum:\n"
                f"{curriculum_list}"
            )},
        ]
        welcome = self._chat(messages, temperature=0.85, num_predict=420)
        self.history.append({"role": "assistant", "content": welcome})
        return welcome

    def _onboarding_turn(self, user_text: str) -> str:
        tnorm = _norm(user_text)
        lesson_cmd, unit_cmd, start_flag = parse_lesson_or_unit_command(user_text)
        if start_flag:
            self._dbg("[ONBOARDING] start-from-beginning -> 1.1")
            self._enter_lesson("1.1")
        elif unit_cmd is not None:
            chosen = first_lesson_of_unit(unit_cmd)
            self._dbg(f"[ONBOARDING] unit {unit_cmd} -> {chosen}")
            self._enter_lesson(chosen)
        elif lesson_cmd is not None:
            self._dbg(f"[ONBOARDING] chapter {lesson_cmd}")
            self._enter_lesson(lesson_cmd)

        if self.session_phase == "ONBOARDING":
            if re.search(r"\bnext\b", tnorm) and ("lesson" in tnorm or "chapter" in tnorm or tnorm == "next"):
                self._dbg("[ONBOARDING] next requested -> 1.1")
                self._enter_lesson("1.1")

        if self.session_phase == "ONBOARDING":
            curriculum_list = curriculum_text_for_welcome()
            messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT_ONBOARDING}]
            messages.extend(self.history)
            messages.append({"role": "user", "content": (
                f"Learner said: {user_text}\n\n"
                "If the learner gave a name, acknowledge it.\n"
                "If they did not choose a Unit yet, ask them to choose Unit 1-5.\n"
                "Re-state briefly how to advance: answer questions to demonstrate mastery or say 'next chapter'.\n"
                "Show the UNIT NAMES list again (exactly), but keep your message short.\n\n"
                "Curriculum:\n"
                f"{curriculum_list}"
            )})
            answer = self._chat(messages, temperature=0.7, num_predict=260)
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": answer})
            return answer
        return ""

    
    def _apply_assessor_turn(self, label: str, momentum_delta: int):
        """
        Apply assessor's turn label + momentum update.
        - momentum_delta is expected in [-3, +3]. ACK should normally be 0.
        - Momentum is global (cross-chapter) and is NOT reset on chapter changes.
        """
        label = (label or "").upper().strip() or "N/A"
        self.last_turn_label = label

        # Clamp delta and update momentum


        # Progress framing: accumulate positive deltas as 'progress points'
        # (ACK typically yields 0; incorrect/confused yields <=0)
        try:
            _tmp_d = int(momentum_delta)
        except Exception:
            _tmp_d = 0
        if _tmp_d > 0:
            self.progress_global += _tmp_d
            self.chapter_progress += _tmp_d

        # Clamp delta and update momentum
        try:
            d = int(momentum_delta)
        except Exception:
            d = 0
        if d > 3:
            d = 3
        if d < -3:
            d = -3

        self.momentum_global += d
        # Clamp momentum to avoid runaway drift
        if self.momentum_global > 10:
            self.momentum_global = 10
        if self.momentum_global < -10:
            self.momentum_global = -10

    def _assess_full(self, last_question: str, last_answer: str) -> Dict[str, Any]:
        item = get_item(self.current_lesson_id)
        outline = "\n".join([f"- {x}" for x in item["outline"]])
        transcript = self._chapter_transcript_text()
        prompt = ASSESSOR_PROMPT_FULL.format(
            teaching_style=self._teaching_style_mode(),
            goal=item["learning_objective"],
            outline=outline,
            last_question=last_question or "",
            last_answer=last_answer or "",
            transcript=transcript or "",
)
        model = self.assessor_model or self.model
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict JSON generator."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.0, "num_predict": 320},
        )
        raw = (resp["message"]["content"] or "").strip()
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return {
                "advance": False,
                "chapter_confidence": 0.0,
                "chapter_reason": "no json",
                "last_turn_label": "N/A",
                "last_turn_confidence": 0.0,
                "support_needed": "N/A",
                "turn_reason": raw[:200],
                "_raw": raw,
            }
        try:
            json_str = m.group(0)
            # Tolerate common model 'almost-JSON' issues (e.g., +3) by sanitizing.
            json_str = re.sub(r'(:\s*)\+(\d)', r'\1\2', json_str)
            data = json.loads(json_str)
            data["_raw"] = json_str
            return data
        except Exception as e:
            return {
                "advance": False,
                "chapter_confidence": 0.0,
                "chapter_reason": f"json parse error: {e}",
                "last_turn_label": "N/A",
                "last_turn_confidence": 0.0,
                "support_needed": "N/A",
                "turn_reason": raw[:200],
                "_raw": raw,
            }

    def turn(self, user_text: str, top_k: int = 4) -> str:
        self.turn_count += 1

        # If this learner message is primarily for onboarding (name/unit choice),
        # we should not run the assessor because there is nothing meaningful to assess yet.
        prev_phase = self.session_phase
        if self.session_phase == "ONBOARDING":
            onboarding_reply = self._onboarding_turn(user_text)
            if onboarding_reply:
                return onboarding_reply
        transitioned_from_onboarding = (prev_phase == "ONBOARDING" and self.session_phase == "IN_CHAPTER")

        tnorm = _norm(user_text)
                # Short replies can indicate low engagement; used only for debug at the moment.
        if len(tnorm) <= 10:
            self.short_reply_streak += 1
        else:
            self.short_reply_streak = 0

        self.chapter_history.append({"role": "user", "content": user_text})
        if self.session_phase == "IN_CHAPTER":
            self.chapter_turns += 1

        lesson_cmd, unit_cmd, start_flag = parse_lesson_or_unit_command(user_text)
        if start_flag:
            self._dbg("[CURRICULUM] start-from-beginning -> 1.1")
            self._enter_lesson("1.1")
        elif unit_cmd is not None:
            chosen = first_lesson_of_unit(unit_cmd)
            self._dbg(f"[CURRICULUM] unit {unit_cmd} -> {chosen}")
            self._enter_lesson(chosen)
        elif lesson_cmd is not None:
            self._dbg(f"[CURRICULUM] chapter -> {lesson_cmd}")
            self._enter_lesson(lesson_cmd)

        if re.search(r"\bnext\b", tnorm) and (("lesson" in tnorm) or ("chapter" in tnorm) or tnorm == "next"):
            nxt = next_lesson_id(self.current_lesson_id)
            if nxt != self.current_lesson_id:
                self._dbg(f"[CURRICULUM] next -> {self.current_lesson_id} -> {nxt}")
                self._enter_lesson(nxt)


        if (
            self.use_assessor
            and self.session_phase == "IN_CHAPTER"
            and not transitioned_from_onboarding
            and len(tnorm) >= 1
            and any(m.get("role") == "assistant" for m in self.chapter_history)
        ):
            last_q = _extract_last_tutor_question(self.history, max_scan=12)
            data = self._assess_full(last_question=last_q, last_answer=user_text)

            adv = bool(data.get("advance", False))
            chap_conf = float(data.get("chapter_confidence", 0.0))
            chap_reason = str(data.get("chapter_reason", ""))
            self.last_chapter_reason = chap_reason
            self.last_chapter_confidence = chap_conf
            label = str(data.get("last_turn_label", "N/A"))
            label_conf = float(data.get("last_turn_confidence", 0.0))
            support = str(data.get("support_needed", "N/A"))
            turn_reason = str(data.get("turn_reason", ""))

            # Expose assessor's turn-level reasoning to the tutor only when confidence is high.
            self.last_turn_confidence = label_conf
            self.last_turn_reason = turn_reason

            delta = int(data.get("momentum_delta", 0)) if str(data.get("momentum_delta", "")).strip() != "" else 0
            # Be extra conservative with negative momentum updates in EASY mode.
            if self._teaching_style_mode() == "EASY" and delta < -1:
                delta = -1

            # ENGAGEMENT OVERRIDE (quick calibration):
            # Treat acknowledgements, reactions, and learner questions as engagement (ACK, delta=0),
            # unless the assessor is very confident it was incorrect/off-topic.
            def _is_ack_like(t: str) -> bool:
                tn = _norm(t)
                if not tn:
                    return True
                ack_set = {
                    'ok', 'okay', 'okey', 'thanks', 'thank you', 'thx', 'merci',
                    'great', 'nice', 'good', 'cool', 'wow', 'alright', 'sure',
                    'fine', 'i see', 'understood'
                }
                if tn in ack_set:
                    return True
                if len(tn) <= 4:
                    return True
                return False

            is_question = '?' in (user_text or '')
            if label_conf < 0.8 and (is_question or _is_ack_like(user_text)):
                label = 'ACK'
                support = 'NONE'
                delta = 0
                turn_reason = 'Learner engagement (acknowledgement/reaction/question) — not graded as an exam answer.'

            self.support_needed = support
            self._apply_assessor_turn(label, momentum_delta=delta)

            self._assess_dbg("\n=========== ASSESSOR DEBUG ===========")
            self._assess_dbg(f"Chapter: {self.current_lesson_id} | advance_threshold={self.advance_conf_threshold:.2f}")
            self._assess_dbg(f"[chapter] advance={adv} conf={chap_conf:.2f} reason={chap_reason}")
            self._assess_dbg(f"[turn] label={label} conf={label_conf:.2f} support_needed={support} delta={delta} reason={turn_reason}")
            self._assess_dbg(f"[global momentum] momentum={self.momentum_global} teaching_style={self._teaching_style_mode()}")
            tx = self._chapter_transcript_text(max_turns=12, max_chars=1400)
            self._assess_dbg("[assessment window excerpt]")
            self._assess_dbg(tx if tx else "(empty)")
            self._assess_dbg("[assessor raw json]")
            self._assess_dbg(str(data.get("_raw", ""))[:1600] or "(none)")
            self._assess_dbg("======================================\n")

            if adv and not (chap_conf >= self.advance_conf_threshold and self.chapter_turns >= 3 and self._chapter_has_min_exchange()):
                self._dbg(f"[ADVANCE-HOLD] assessor advance=true but gate blocked: conf={chap_conf:.2f} turns={self.chapter_turns} min_exchange={self._chapter_has_min_exchange()}")

            if adv and chap_conf >= self.advance_conf_threshold and self.chapter_turns >= 3 and self._chapter_has_min_exchange():
                nxt = next_lesson_id(self.current_lesson_id)
                if nxt != self.current_lesson_id:
                    self._dbg(f"[CURRICULUM] advancing {self.current_lesson_id} -> {nxt}")
                    self._enter_lesson(nxt)


        retrieved: List[Dict[str, Any]] = []
        notes: Optional[str] = None
        best_score = -1.0

        if self.lesson_phase == "INTRO":
            item = get_item(self.current_lesson_id)
            teach_query = f"{item.get('chapter', item.get('lesson_title'))}. Outline: " + "; ".join(item["outline"])
            retrieved = self.retrieve_fn(teach_query, top_k=top_k)
            notes, best_score = build_optional_notes(retrieved)
            self.last_best_score = best_score
        else:
            item = get_item(self.current_lesson_id)
            anchor = item.get("chapter", item.get("lesson_title"))
            anchored_query = f"{anchor}. Learner: {user_text}"
            retrieved = self.retrieve_fn(anchored_query, top_k=top_k)
            notes, best_score = build_optional_notes(retrieved)
            self.last_best_score = best_score

        self._dbg("\n================ DEBUG ================")
        self._dbg(f"Turn: {self.turn_count} | short_reply_streak={self.short_reply_streak} | move_cycle={self.move_cycle}")
        self._dbg(f"Strategy: {self.last_strategy}")
        self._dbg(f"Session phase: {self.session_phase} | Chapter: {self.current_lesson_id} | phase={self.lesson_phase}")
        self._dbg(f"Global momentum: momentum={self.momentum_global} | teaching_style={self._teaching_style_mode()}")
        self._dbg(f"Last assessor label: {self.last_turn_label} | support_needed={self.support_needed}")
        if best_score != -1.0:
            self._dbg(f"Retrieval: n={len(retrieved)} best_score={best_score:.4f}")
            # Retrieval transparency: show retrieved chunk ids + top chunk excerpt
            try:
                ids_scores = [f"{r.get('id')}@{float(r.get('score', -1.0)):.3f}" for r in retrieved]
            except Exception:
                ids_scores = []
            self._dbg(f"Retrieved chunks: {', '.join(ids_scores) if ids_scores else '(none)'}")
            if retrieved:
                # Use first item as top chunk for excerpt display (already top-k ordered)
                top_txt = (retrieved[0].get('text') or '').strip().replace('\n', ' ')
                if len(top_txt) > 260:
                    top_txt = top_txt[:260].rstrip() + '...'
                self._dbg(f"Top chunk excerpt: {top_txt if top_txt else '(empty)'}")
        else:
            self._dbg("Retrieval skipped")
        self._dbg("======================================\n")

        messages: List[Dict[str, str]] = [{"role": "system", "content": self._system_prompt()}]
        messages.extend(self.history)
        if notes:
            messages.append({"role": "system", "content": notes})

        messages.append({"role": "system", "content": (
            "Retention/engagement directives:\n"
            f"- Teaching style mode: {self._teaching_style_mode()}\n"
            f"- Last turn label: {self.last_turn_label}\n"
            f"- Support needed: {self.support_needed}\n"
            "- Always include a brief progress signal when natural (1 sentence).\n"
            "- Give diagnostic feedback tied to the learner's last reply when possible.\n"
            "- Keep to at most ONE question at the end of your reply.\n"
        )})

        if self.support_needed == "HINT":
            self.last_strategy = "SUPPORT_HINT"
            messages.append({"role": "system", "content": "Give a short hint (not the full answer), then ask ONE open check question."})
        elif self.support_needed == "GUIDED_STEPS":
            self.last_strategy = "SUPPORT_GUIDED_STEPS"
            messages.append({"role": "system", "content": "Guide the learner with a few steps, then ask ONE open check question."})
        elif self.support_needed == "EXAMPLE":
            self.last_strategy = "SUPPORT_EXAMPLE"
            messages.append({"role": "system", "content": "Give one clear example, then ask ONE open check question."})

        if self.lesson_phase == "INTRO":
            self.last_strategy = "LESSON_INTRO"
            messages.append({"role": "system", "content": (
                "You are starting (or restarting) the current chapter. "
                "Write a lesson in 3–6 short paragraphs that covers the outline points in a clear order. "
                "Do NOT ask any question until after those paragraphs. "
                "Then ask exactly ONE question that checks the learning objective (it can be open-ended OR a small multiple-choice/recognition check, depending on teaching style)."
            )})
            self.lesson_phase = "ASK"
        else:

            # Pedagogical move policy: keep the session alive with varied interaction.
            # We rotate between: CHECK_IN, INVITE_QUESTION, MICRO_PRACTICE, EXAMPLE, SUMMARY.
            self.move_cycle = (self.move_cycle + 1) % 5
            move_name = ["INVITE_QUESTION", "EXAMPLE", "MICRO_PRACTICE", "CHECK_IN", "SUMMARY"][self.move_cycle]
            self._dbg(f"[MOVE] cycle={self.move_cycle} move={move_name}")
            self.last_strategy = f"MOVE_{move_name}"
            # Always end with ONE clear CTA question (prevents cold endings and short 'ok' replies).
            CTA_BY_MOVE = {
                "CHECK_IN": "Does that make sense so far, or which part feels unclear?",
                "INVITE_QUESTION": "What question would you like to ask me about this chapter?",
                "MICRO_PRACTICE": "Quick practice: which option is correct (A, B, or C)?",
                "EXAMPLE": "Come up with an example pertaining to the current chapter",
                "SUMMARY": "In one sentence, what key idea are you taking away from this chapter?",
            }
            cta = CTA_BY_MOVE.get(move_name, "What part would you like to focus on next?")

            # Minimal move guidance (keep prompt small) + enforce a CTA.
            messages.append({"role": "system", "content": (
                f"Pedagogical move for this turn: {move_name}. "
                "Do NOT end cold. End with exactly ONE CTA question: "
                f"\"{cta}\" "
                "If support_needed is set (HINT/GUIDED_STEPS/EXAMPLE), prioritize that support."
            )})
        messages.append({"role": "user", "content": user_text})
        answer = self._enforce_header(self._chat(messages))
        #answer = _enforce_single_question(answer)

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": answer})
        self.chapter_history.append({"role": "assistant", "content": answer})
        return answer
