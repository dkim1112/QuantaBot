import tiktoken

def estimate_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def calculate_optimal_batch(summaries: list, estimate_tokens_func) -> int:
    MAX_TOKENS_PER_REQUEST = 4096
    TARGET_TOKENS_PER_BATCH = MAX_TOKENS_PER_REQUEST * 0.6

    sample_size = min(len(summaries), 10)
    sample_summaries = summaries[:sample_size]

    total_tokens = sum(
        estimate_tokens_func(summary) for summary in sample_summaries
    )
    avg_tokens_per_summary = total_tokens / sample_size

    optimal_size = max(1, int(TARGET_TOKENS_PER_BATCH / avg_tokens_per_summary))
    return min(max(optimal_size, 2), 20)  # Between 2 and 20 summaries per batch
