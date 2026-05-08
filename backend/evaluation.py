from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EvalSample:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


def run_ragas_evaluation(samples: List[EvalSample], provider: str = "groq") -> list[dict]:
    """
    Run Ragas Faithfulness + ResponseRelevancy on Q&A samples.

    Faithfulness   — Is the answer grounded in the retrieved context?
    ResponseRelevancy — Is the answer relevant to the question?
    """
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # ── LLM for evaluation ────────────────────────────────────────────────
    if provider == "groq":
        from langchain_groq import ChatGroq
        from config.settings import GROQ_API_KEY, GROQ_FAST_MODEL
        llm = LangchainLLMWrapper(ChatGroq(api_key=GROQ_API_KEY, model=GROQ_FAST_MODEL))
    else:
        from langchain_anthropic import ChatAnthropic
        from config.settings import ANTHROPIC_API_KEY, ANTHROPIC_FAST_MODEL
        llm = LangchainLLMWrapper(
            ChatAnthropic(api_key=ANTHROPIC_API_KEY, model=ANTHROPIC_FAST_MODEL)
        )

    # ── Embeddings (reuse local model) ────────────────────────────────────
    from langchain_community.embeddings import HuggingFaceEmbeddings
    emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    ragas_samples = [
        SingleTurnSample(
            user_input=s.question,
            retrieved_contexts=s.contexts if s.contexts else ["No context retrieved"],
            response=s.answer,
            reference=s.ground_truth,
        )
        for s in samples
    ]

    dataset = EvaluationDataset(samples=ragas_samples)
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=llm),
            ResponseRelevancy(llm=llm, embeddings=emb),
        ],
    )
    return result.to_pandas().to_dict("records")
