"""
Labelled evaluation dataset for RAGAS quality benchmarking.

Each entry is a (question, reference_answer) pair derived from the arXiv
corpus indexed in ChromaDB. Reference answers were written by hand from
the original papers — these are the 'ground truth' to compare system
answers against.

Kept small (15 Q/A pairs) to respect Groq's free-tier daily token quota
while still giving statistically-meaningful metric averages.
"""

EVAL_SET = [
    {
        "question": "What is Retrieval-Augmented Generation (RAG)?",
        "reference": (
            "RAG is an architecture that combines a parametric language "
            "model with a non-parametric memory (a retriever over an "
            "external corpus). Given a query, the retriever fetches "
            "relevant documents and the generator conditions on them to "
            "produce the output, improving factual accuracy over a pure "
            "LM baseline."
        ),
    },
    {
        "question": "What problem does the 'Lost in the Middle' paper identify?",
        "reference": (
            "LLMs reliably use information at the beginning and end of "
            "their input context but often miss information placed in the "
            "middle. Performance on retrieval-style tasks degrades "
            "significantly when the answer is in the middle of a long "
            "context window."
        ),
    },
    {
        "question": "What is Dense Passage Retrieval (DPR)?",
        "reference": (
            "DPR is a dense retrieval method that learns question and "
            "passage encoders separately, such that relevant pairs have "
            "higher dot-product similarity than irrelevant ones. It is "
            "trained with contrastive learning over in-batch negatives "
            "and typically uses a BERT-based encoder."
        ),
    },
    {
        "question": "How does Self-RAG differ from standard RAG?",
        "reference": (
            "Self-RAG teaches the model to generate special reflection "
            "tokens that decide when to retrieve, evaluate whether "
            "retrieved passages are relevant, and assess whether the "
            "generated answer is supported by evidence. It selectively "
            "retrieves on demand rather than always retrieving."
        ),
    },
    {
        "question": "What is Corrective RAG (CRAG)?",
        "reference": (
            "CRAG adds a lightweight retrieval evaluator that grades "
            "retrieved documents as correct, incorrect, or ambiguous. "
            "For incorrect or ambiguous cases it triggers a corrective "
            "action such as web search, improving robustness when the "
            "initial retrieval is poor."
        ),
    },
    {
        "question": "What does the RAGAS framework measure?",
        "reference": (
            "RAGAS is a reference-free evaluation framework for RAG "
            "pipelines. Its core metrics are faithfulness (how grounded "
            "the answer is in the retrieved context), answer relevancy "
            "(how well the answer addresses the question), context "
            "precision, and context recall."
        ),
    },
    {
        "question": "What is Chain-of-Thought prompting?",
        "reference": (
            "Chain-of-Thought prompting asks the language model to "
            "produce intermediate reasoning steps before the final "
            "answer. It substantially improves performance on "
            "arithmetic, commonsense, and symbolic reasoning tasks, "
            "especially for large models."
        ),
    },
    {
        "question": "What is Zero-shot Chain-of-Thought?",
        "reference": (
            "Zero-shot CoT is the finding that simply appending a "
            "trigger like 'Let's think step by step' to a prompt "
            "elicits multi-step reasoning from large language models "
            "without any worked examples, substantially improving "
            "reasoning benchmark scores."
        ),
    },
    {
        "question": "What is Tree of Thoughts?",
        "reference": (
            "Tree of Thoughts generalises Chain-of-Thought by exploring "
            "multiple reasoning branches, evaluating intermediate states, "
            "and using search strategies such as breadth-first or "
            "depth-first to find a solution. It outperforms CoT on "
            "problems that benefit from lookahead or backtracking."
        ),
    },
    {
        "question": "What is ReAct?",
        "reference": (
            "ReAct interleaves reasoning (Thought) and acting (Action) "
            "steps in language model agents. The model alternates between "
            "thinking about what to do next and taking actions (like "
            "calling a search tool), which improves task performance and "
            "interpretability over reasoning-only or acting-only baselines."
        ),
    },
    {
        "question": "What is Toolformer?",
        "reference": (
            "Toolformer is a self-supervised approach that teaches "
            "language models to decide which APIs to call, when to call "
            "them, what arguments to pass, and how to incorporate the "
            "results into text generation. Training data is generated by "
            "the model itself via a self-supervised filtering procedure."
        ),
    },
    {
        "question": "What is HuggingGPT?",
        "reference": (
            "HuggingGPT uses a large language model (ChatGPT) as a "
            "controller that plans tasks, selects appropriate expert "
            "models from the Hugging Face Hub, executes them, and "
            "aggregates their outputs. It enables language, vision, "
            "speech, and other cross-modal tasks via model composition."
        ),
    },
    {
        "question": "What is the AgentBench benchmark?",
        "reference": (
            "AgentBench is a multi-dimensional benchmark that evaluates "
            "LLMs as agents across eight distinct environments, including "
            "operating systems, databases, knowledge graphs, card games, "
            "and web browsing. It tests multi-turn, open-ended decision "
            "making rather than single-turn QA."
        ),
    },
    {
        "question": "What is InstructGPT and how is it trained?",
        "reference": (
            "InstructGPT is a GPT-3 model fine-tuned to follow "
            "instructions using reinforcement learning from human "
            "feedback (RLHF). The pipeline has three stages: supervised "
            "fine-tuning on human demonstrations, training a reward model "
            "on human preference comparisons, and PPO optimisation against "
            "the reward model."
        ),
    },
    {
        "question": "What is the MMLU benchmark?",
        "reference": (
            "MMLU (Massive Multitask Language Understanding) is a "
            "benchmark of multiple-choice questions across 57 subjects "
            "including STEM, humanities, and social sciences. It measures "
            "broad world knowledge and problem-solving ability at roughly "
            "high-school to professional difficulty."
        ),
    },
]
