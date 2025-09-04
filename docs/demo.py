from demo_lib import get_llm, retrieve_top_k, ChatSession

def load_docs():
    return {
        "sample_policy": open("data/context/sample_policy.txt").read(),
        "sample_corpus": open("data/context/sample_corpus.txt").read()
    }

def run_demo():
    llm = get_llm()
    print("== Prompt Template Library (generic) ==")
    print("Persona: Compliance-aware Prompt Engineer explaining RAG.\n")

    print("== Multi-turn Conversation ==")
    chat = ChatSession(system="You are a careful assistant.")
    chat.add_user("Explain RAG briefly.")
    ans1 = llm.invoke(chat.prompt_text()); chat.add_ai(ans1)
    print("AI:", ans1)
    chat.add_user("Now apply it to summarizing client-facing reports with safety rules.")
    ans2 = llm.invoke(chat.prompt_text()); chat.add_ai(ans2)
    print("AI:", ans2, "\n")

    print("== Tiny RAG Demo (generic corpus) ==")
    docs = load_docs()
    q = "How should we handle uncertain answers in client communications?"
    hits = retrieve_top_k(q, docs, k=2)
    context = "\n---\n".join([h[1] for h in hits])
    prompt = (
        "Answer using ONLY this context. If it's not covered, say 'Not in context.'\n"
        f"Context:\n{context}\n\nQuestion: {q}"
    )
    print("Answer:", llm.invoke(prompt))

if __name__ == "__main__":
    run_demo()
