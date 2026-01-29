## Design Explanations 

### Chunk size choice
I used 512-word chunks with 50-word overlap. This keeps each chunk semantically coherent (full paragraphs) while still allowing fine-grained retrieval and avoids splitting important sentences across chunks.

### Observed retrieval failure case
A failure case occurs with questions that require combining facts from multiple chunks, for example: "What was the revenue growth compared to last year?". If one chunk has "Revenue was 10M in 2022" and another has "Revenue was 12M in 2023", retrieval may fetch only one chunk, so the model cannot compute the growth correctly.

### Metric tracked
The `/query` endpoint returns `latency_ms`, which measures end-to-end latency: embedding the question, FAISS similarity search, and LLM generation. This helps monitor and optimize system performance.
