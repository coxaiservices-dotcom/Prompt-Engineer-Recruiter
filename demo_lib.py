import os, re
from typing import List, Dict, Tuple

class MockLLM:
    def __init__(self, name="mock-llm"):
        self.name = name
    def invoke(self, prompt: str) -> str:
        if "RAG" in prompt or "retriev" in prompt.lower():
            return "RAG uses retrieved context to ground answers; if info is missing, say 'Not in context.'"
        if "prompt library" in prompt.lower() or "template" in prompt.lower():
            return "Loaded templates across Structuring, Safety, Creativity. Select one to render."
        if "policy" in prompt.lower():
            return "Per policy: avoid advice, list risks, keep tone professional, escalate when unsure."
        return "Here is a concise, policy-aware answer based on your prompt."

def get_llm():
    # Offline-first demo; returns MockLLM to avoid network dependence during interviews
    return MockLLM()

def simple_chunk(text: str, max_chars: int = 420) -> List[str]:
    chunks, buf = [], ""
    for line in text.splitlines():
        if len(buf) + len(line) + 1 > max_chars:
            if buf.strip(): chunks.append(buf.strip())
            buf = line + "\n"
        else:
            buf += line + "\n"
    if buf.strip(): chunks.append(buf.strip())
    return chunks

def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", s.lower())

def score_overlap(query: str, chunk: str) -> int:
    q = set(tokenize(query))
    c = set(tokenize(chunk))
    return len(q & c)

def retrieve_top_k(query: str, docs: Dict[str, str], k: int = 3) -> List[Tuple[str, str, int]]:
    scored = []
    for name, text in docs.items():
        for i, ch in enumerate(simple_chunk(text)):
            s = score_overlap(query, ch)
            if s > 0:
                scored.append((f"{name}#chunk{i+1}", ch, s))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:k]

class ChatSession:
    def __init__(self, system="You are a careful, policy-aware assistant."):
        self.system = system
        self.history: List[Tuple[str, str]] = []
    def add_user(self, msg: str): self.history.append(("user", msg))
    def add_ai(self, msg: str): self.history.append(("assistant", msg))
    def prompt_text(self) -> str:
        lines = [f"System: {self.system}"]
        for role, content in self.history:
            lines.append(f"{role.title()}: {content}")
        return "\n".join(lines)
