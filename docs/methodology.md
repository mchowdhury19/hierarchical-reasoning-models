\# Research Methodology



\## Research Question



\*\*Can models trained with single-step supervision on optimal trajectories reliably perform multi-step autoregressive reasoning?\*\*



---



\## Hypothesis



Models trained to predict the next optimal action given the current state will achieve high training accuracy but fail catastrophically during autoregressive deployment due to distribution shift between training contexts (optimal states) and inference contexts (model's own predictions).



This is the "Illusion of Thinking" - models appear to learn during training but cannot reason during deployment.



---



\## Experimental Design



\### Domain: Block World Puzzles



\*\*Task:\*\* Rearrange blocks on pegs to match target configuration.



\*\*Complexity Levels:\*\*

\- N = number of blocks (1-10)

\- Training: N=1-7 (3,763 samples from 549 puzzles)

\- Testing: N=8-10 (2,913 samples from 300 puzzles)



\*\*Why Block World:\*\*

\- Well-defined optimal solutions

\- Clear hierarchical structure (blocks must be moved in order)

\- Scalable complexity

\- Simple enough to generate large datasets

\- Complex enough to challenge reasoning



\### Models Tested



\*\*Small Models (trained from scratch):\*\*

1\. Elman RNN (0.47M params) - Vanilla recurrent

2\. LSTM (1.85M params) - Gated recurrent

3\. GNN (70K params) - Graph convolutional

4\. Custom Transformer (0.8M params) - Attention mechanism



\*\*Large Models:\*\*

5\. T5 from scratch (223M params) - No pre-training

6\. GPT-2 (355M params) - Pre-trained, fine-tuned

7\. T5 Symbolic (223M params) - Pre-trained, fine-tuned

8\. T5 Natural (223M params) - Pre-trained, fine-tuned



\*\*Representations Tested:\*\*

\- Symbolic tokens (for RNN/LSTM/Transformer/GPT-2)

\- Natural language (for T5)

\- Graph structure (for GNN)



---



\## Training Protocol



\### Single-Step Supervision



\*\*Objective:\*\* Predict optimal next action given current state.



\*\*Loss Function:\*\* Cross-entropy over action classes.



\*\*Key Characteristic:\*\* Model trained on optimal states from expert trajectories.



\### Configuration



\*\*Small Models:\*\*

\- Epochs: 20

\- Learning rate: 3e-4

\- Optimizer: AdamW

\- Batch size: 32



\*\*Large Pre-trained Models:\*\*

\- Epochs: 3 (fewer due to pre-training)

\- Learning rate: 5e-5 (lower to preserve pre-training)

\- Optimizer: AdamW

\- Batch size: 8 (smaller due to model size)



---



\## Evaluation Metrics



\### 1. Teacher Forcing Accuracy (TF)



\*\*What it measures:\*\* Single-step prediction accuracy on optimal states.



\*\*Procedure:\*\*

\- Provide model with correct current state

\- Model predicts next action

\- Compare with optimal action

\- Metric: Percentage of correct predictions



\*\*What it reveals:\*\* How well model learned patterns in training distribution.



\### 2. Autoregressive In-Distribution (ID)



\*\*What it measures:\*\* Full puzzle solving on training complexity (N=1-7).



\*\*Procedure:\*\*

\- Start with puzzle initial state

\- Model predicts action

\- Apply action to get next state (may be imperfect)

\- Repeat until goal reached or failure detected

\- Metric: Percentage of puzzles solved



\*\*What it reveals:\*\* Can model handle its own predictions?



\### 3. Autoregressive Out-of-Distribution (OOD)



\*\*What it measures:\*\* Full puzzle solving on larger problems (N=8-10).



\*\*Procedure:\*\* Same as ID but on larger, never-seen puzzle sizes.



\*\*What it reveals:\*\* Does model generalize to increased complexity?



---



\## Critical Innovation: Gap Analysis



\*\*The key insight comes from comparing metrics:\*\*

```

Gap = Teacher Forcing Accuracy - Autoregressive Accuracy

```



\*\*Large gap indicates:\*\*

\- Model learned patterns on optimal states

\- Model cannot handle distribution shift to own predictions

\- "Illusion of Thinking" - appears capable but isn't



\*\*Small gap indicates:\*\*

\- Model has robust representations

\- Model handles distribution shift well

\- Genuine reasoning capability



---



\## Failure Mode Detection



During autoregressive evaluation, we systematically detect and categorize failures:



\### 1. Mode Collapse

Model outputs same action repeatedly, ignoring input state.



\### 2. Oscillation

Model cycles between same 2-3 states without progress.



\### 3. Invalid Actions

Model predicts illegal moves (wrong block, empty peg, etc.).



\### 4. Timeout

Model exceeds maximum steps without solving.



\*\*Purpose:\*\* Understand HOW models fail, not just that they fail.



---



\## Statistical Analysis



\### Breakdown by Complexity

Results analyzed separately for each N (1-10) to identify scaling behavior.



\### Failure Type Distribution

Quantify prevalence of each failure mode per architecture.



\### Trajectory Analysis

Examine step-by-step reasoning to identify where models go wrong.



---



\## Experimental Controls



\### Data Consistency

All models trained on identical datasets with identical train/test splits.



\### Evaluation Consistency

All models evaluated using same autoregressive protocol with same failure detection.



\### Random Seeds

Fixed seeds for reproducibility.



\### Hardware

All experiments on same GPU type (NVIDIA A100/L4/T4).



---



\## Why This Methodology Reveals the Problem



\*\*Standard evaluation (teacher forcing) would miss the problem:\*\*

\- Shows high accuracy

\- Suggests models learned successfully

\- Provides false confidence



\*\*Our evaluation (autoregressive) reveals the problem:\*\*

\- Shows catastrophic failure

\- Demonstrates distribution shift issue

\- Exposes gap between training and deployment



\*\*This is critical for AI safety:\*\* We cannot trust training metrics alone. We must evaluate in deployment conditions.



---



\## Limitations



\### Domain Specificity

Results are from Block World. Generalization to other domains requires additional experiments (Tower of Hanoi, River Crossing, Checkers Jumping planned).



\### Optimal Trajectories Only

Training data contains only optimal solutions. Real-world data may include suboptimal trajectories.



\### Single-Step Supervision

We test this specific training paradigm. Other training methods (RL, process supervision) may perform differently.



\### Computational Constraints

Limited by available compute - could not test all possible architectures, hyperparameters, or model sizes.



---



\## Implications for AI Safety



This methodology demonstrates:



1\. \*\*Standard metrics are insufficient\*\* - Need deployment-realistic evaluation

2\. \*\*Distribution shift is critical\*\* - Training conditions ≠ deployment conditions

3\. \*\*Scale thresholds exist\*\* - Gradual improvement does NOT occur, sharp transitions do

4\. \*\*Pre-training is essential\*\* - Architecture and size alone insufficient



For high-stakes applications (education, healthcare), we must:

\- Evaluate autoregressively, not just teacher forcing

\- Test out-of-distribution generalization

\- Understand failure modes systematically

\- Require mechanistic interpretability before deployment

