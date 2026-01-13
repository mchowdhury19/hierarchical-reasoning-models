\# Investigating the "Illusion of Thinking" in Hierarchical Reasoning Models



\## Overview



Systematic empirical investigation of failure modes in language models trained with single-step supervision on hierarchical reasoning tasks. This research demonstrates that models can achieve high training accuracy while failing catastrophically during autoregressive deployment, revealing what we term the "confident false negative" problem.



\*\*Key Finding:\*\* Models output intermediate steps that appear correct to human evaluators but contain systematic errors in internal representations that compound during multi-step reasoning. This has critical implications for AI safety in high-stakes applications (education, healthcare) where we cannot verify correctness without understanding what happens at each internal layer.



---



\## Research Question



\*\*Can models trained with single-step supervision on optimal trajectories reliably perform multi-step autoregressive reasoning?\*\*



\*\*Answer:\*\* No, not for models under 200M parameters. Pre-training is essential.



---



\## Models Evaluated



| Model | Parameters | Pre-trained? | Architecture | Teacher Forcing | Auto ID (N=1-7) | Auto OOD (N=8-10) |

|-------|-----------|--------------|--------------|-----------------|-----------------|-------------------|

| Elman RNN | 0.47M | No | Recurrent | 0.55% | 0.36% | 0.00% |

| LSTM | 1.85M | No | Gated RNN | 2.99% | 1.28% | 0.00% |

| \*\*GNN (Graph)\*\* | \*\*0.07M\*\* | \*\*No\*\* | \*\*Graph Conv\*\* | \*\*10.85%\*\* | \*\*1.09%\*\* | \*\*0.00%\*\* |

| Custom Transformer | 0.8M | No | Attention | 4.01% | 0.55% | 0.00% |

| T5 (from scratch) | 223M | No | Enc-Dec | ~4% | ~0.5% | 0.00% |

| GPT-2 Symbolic | 355M | Yes | Decoder | 99.47% | 98.91% | 49.67% |

| T5 Symbolic | 223M | Yes | Enc-Dec | 97.87% | 97.45% | 80.33% |

| T5 Natural Language | 223M | Yes | Enc-Dec | 99.26% | 99.09% | 85.67% |



\*\*Total: 8 configurations across 5 architectures\*\*



---



\## Critical Findings



\### 1. The "Confident False Negative" Problem



Models generate outputs that appear plausible at each intermediate step but contain systematic errors in internal representations. External observers (including trained evaluators) cannot distinguish between "actually correct reasoning" and "convincingly wrong reasoning" without understanding internal model states.



\*\*Example:\*\*

```

State 1 → Predicted Action: Move A from peg 1 to peg 0 (appears correct)

State 2 → Predicted Action: Move B from peg 2 to peg 1 (appears correct)

State 3 → Predicted Action: STOP (appears correct if goal appears reached)

Result: Failed - goal was not actually achieved



Problem: Each step looked reasonable, but internal goal representation was flawed from the start.

```



\### 2. Scale Threshold



\*\*Critical gap at ~200M parameters:\*\*

\- Models < 2M params: 0.36% - 1.28% autoregressive success (catastrophic failure)

\- Models > 200M params (pre-trained): 98.91% - 99.09% success



\*\*T5 from scratch (223M params) proves scale alone insufficient:\*\*

\- Without pre-training: ~4% teacher forcing, ~0.5% autoregressive

\- With pre-training: 99.26% teacher forcing, 99.09% autoregressive

\- \*\*Conclusion: Pre-trained representations are essential, not just parameter count\*\*



\### 3. Architecture Efficiency



\*\*Most parameter-efficient: GNN (70K parameters)\*\*

\- Achieves 10.85% teacher forcing accuracy

\- 26x fewer parameters than LSTM (1.85M)

\- Graph structure most effective representation for small models

\- Still fails autoregressively (1.09%) - architecture alone insufficient



\### 4. Single-Step Metrics are Misleading



\*\*Teacher forcing accuracy does NOT predict deployment performance:\*\*

\- Custom Transformer: 4.01% → 0.55% (3.46pp gap)

\- GNN: 10.85% → 1.09% (9.76pp gap)

\- LSTM: 2.99% → 1.28% (1.71pp gap)



This is the core "Illusion of Thinking" - models appear to learn during training but cannot reason during deployment.



\### 5. Failure Modes



\*\*Pure Mode Collapse (Elman RNN):\*\*

\- Outputs same action for ALL inputs

\- Example: Always "Move C from peg 1 to peg 2"

\- Cause: Vanishing gradients, no gating



\*\*Partial Mode Collapse (LSTM):\*\*

\- Different fixed action, still ignoring input

\- Gating helps slightly but insufficient



\*\*STOP Bias (GNN):\*\*

\- Predicts STOP 42% of time (should be 10%)

\- Most diverse predictions but still biased



\*\*Oscillation (GPT-2 on OOD):\*\*

\- Cycles between same 2-3 states

\- 80% of OOD failures show this pattern

\- Myopic planning, no global reasoning



---



\## Implications for AI Safety



\### The Black Box Problem



\*\*We fundamentally don't understand what happens at each internal layer.\*\*



For high-stakes applications (AI tutoring children, medical diagnosis assistance), we cannot:

\- Verify reasoning correctness by observing outputs alone

\- Detect when intermediate steps are subtly flawed

\- Trust single-step accuracy metrics for deployment decisions

\- Scale systems safely without mechanistic understanding



\### What This Research Demonstrates



1\. \*\*Evaluation Gap:\*\* Standard metrics (accuracy, loss) provide false confidence

2\. \*\*Distribution Shift:\*\* Training on optimal states != handling model's own predictions

3\. \*\*Internal Opacity:\*\* High-performing models fail in ways we can't detect externally

4\. \*\*Scaling Requirements:\*\* Need both scale (200M+ params) AND pre-training for reliability



\### Research Directions Needed



\- \*\*Mechanistic Interpretability:\*\* Understand internal representations at each reasoning step

\- \*\*Scalable Oversight:\*\* Verify multi-step reasoning we cannot manually check

\- \*\*Process Supervision:\*\* Evaluate reasoning quality, not just final answers

\- \*\*Adversarial Testing:\*\* Systematically find hidden failure modes



---



\## Domain \& Dataset



\*\*Block World Puzzles:\*\*

\- State: Blocks stacked on 3 pegs

\- Goal: Rearrange to target configuration

\- Complexity: N blocks (N=1-10)

\- Training: N=1-7 (3,763 samples, 549 puzzles)

\- Testing: N=8-10 (2,913 samples, 300 puzzles)



\*\*Future Domains (Planned):\*\*

\- Tower of Hanoi

\- River Crossing

\- Checkers Jumping



---



\## Technical Stack



\- \*\*PyTorch\*\* - Model implementation

\- \*\*PyTorch Geometric\*\* - Graph neural networks

\- \*\*HuggingFace Transformers\*\* - T5, GPT-2 fine-tuning

\- \*\*Weights \& Biases\*\* - Experiment tracking

\- \*\*NumPy, Pandas\*\* - Data processing

\- \*\*Matplotlib, Seaborn\*\* - Visualization



---



\## Repository Contents



\### Code Snippets

Sample implementations demonstrating key technical contributions:

\- `code\_snippets/gnn\_architecture.py` - Graph convolutional network (70K params)

\- `code\_snippets/evaluation\_framework.py` - Autoregressive evaluation protocol

\- `code\_snippets/graph\_encoding.py` - State-to-graph representation

\- `code\_snippets/failure\_detection.py` - Mode collapse detection



\### Pseudocode

High-level algorithms for sensitive implementations:

\- `pseudocode/training\_pipeline.md` - Single-step supervision training

\- `pseudocode/t5\_finetuning.md` - Large model fine-tuning approach



\### Documentation

\- `docs/methodology.md` - Experimental design and evaluation protocol

\- `docs/findings.md` - Comprehensive results and analysis

\- `docs/failure\_modes.md` - Detailed failure taxonomy



\*\*Note:\*\* This research is part of an unpublished paper currently in preparation. Full experimental codebase cannot be shared prior to publication. Code snippets and pseudocode demonstrate technical approach without compromising team's intellectual property.



---



\## My Contributions



As part of a 4-person research team:



\*\*Architecture Implementation \& Comparison:\*\*

\- Implemented Graph Neural Network achieving 10.85% accuracy with only 70K parameters

\- Built Elman RNN and LSTM encoder-decoder models from scratch

\- Led systematic comparison across 8 configurations



\*\*Evaluation Framework Design:\*\*

\- Designed autoregressive evaluation protocol revealing the "confident false negative" problem

\- Implemented teacher forcing vs. autoregressive performance comparison

\- Created systematic failure mode detection (mode collapse, oscillation, invalid actions)

\- Developed out-of-distribution generalization testing methodology



\*\*Graph Representation Innovation:\*\*

\- Engineered graph encoding for hierarchical reasoning (nodes=blocks+pegs, edges=relationships)

\- Achieved most parameter-efficient architecture (26x more efficient than LSTM)



\*\*Failure Analysis:\*\*

\- Discovered and documented the "confident false negative" problem

\- Quantified mode collapse patterns across architectures

\- Analyzed STOP bias in GNN predictions

\- Identified oscillation loops in GPT-2 OOD failures



\*\*Team:\*\* Shivank Garg (Lead), \[Your Name], Jason Chen, Gowrav, Chowdhury



---



\## Key Takeaways



1\. \*\*Small models universally fail\*\* (<1.3% autoregressive) despite learning single-step predictions

2\. \*\*GNN most efficient\*\* architecture (10.85% with 70K params) but still fails autoregressively

3\. \*\*Scale + pre-training essential\*\* - T5 from scratch (223M) performs like Custom Transformer (0.8M)

4\. \*\*Cannot trust test metrics\*\* - 99% training accuracy → <1% deployment success possible

5\. \*\*Black box is dangerous\*\* - Need mechanistic interpretability for safe deployment



\*\*For AI in education and healthcare:\*\* We cannot deploy systems we don't understand at every layer. Single-step supervision creates models that fool evaluation metrics while being fundamentally unreliable.



---



\## Research Context



\*\*Duration:\*\* 6 weeks (November 2024 - January 2025)  

\*\*Status:\*\* Manuscript in preparation (planned NeurIPS 2026 workshop submission)  

\*\*Inspired by:\*\* Apple's "GSM-Symbolic" paper on mathematical reasoning limitations



---



\## Contact



\*\*Project Repository:\*\* \[https://github.com/mchowdhury19/hierarchical-reasoning-models](https://github.com/mchowdhury19/hierarchical-reasoning-models)



For questions about methodology, technical implementation, or collaboration opportunities, please reach out.



---



\*This research investigates fundamental questions about AI safety, interpretability, and the gap between training metrics and deployment reliability. Understanding these failure modes is critical for building AI systems that are safe for high-stakes applications.\*

