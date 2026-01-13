\# Failure Modes Analysis



\## The "Confident False Negative" Problem



\*\*Definition:\*\* Models generate outputs that appear plausible at each intermediate step to external observers (including human evaluators) but contain systematic errors in internal representations that compound during multi-step reasoning.



\*\*Why "Confident":\*\* Model assigns high probability to predictions.



\*\*Why "False Negative":\*\* Failure is not detected by standard evaluation metrics.



\*\*Why Critical for AI Safety:\*\* Cannot verify correctness without understanding internal model states.



---



\## Taxonomy of Failure Modes



\### 1. Pure Mode Collapse



\*\*Observed in:\*\* Elman RNN (100% of predictions)



\*\*Symptom:\*\* Model outputs identical action for ALL inputs, completely ignoring state.



\*\*Example:\*\*

```

Input State: \[\['A'], \[], \[]]

Output: "Move C from peg 1 to peg 2"



Input State: \[\[], \['B'], \[]]

Output: "Move C from peg 1 to peg 2"



Input State: \[\[], \[], \['C']]

Output: "Move C from peg 1 to peg 2"

```



\*\*Cause:\*\*

\- Vanishing gradients in vanilla RNN

\- No gating mechanism to preserve information

\- Model learns single dominant pattern from training data

\- Simplest solution: predict most common training action



\*\*Detection:\*\*

\- Check if same action predicted for diverse inputs

\- Measure action diversity: entropy of action distribution

\- Compare prediction distribution to training distribution



\*\*Frequency:\*\* 100% for Elman RNN, 0% for other models



---



\### 2. Partial Mode Collapse



\*\*Observed in:\*\* LSTM (50% of predictions)



\*\*Symptom:\*\* Model outputs same action frequently but not always.



\*\*Example:\*\*

```

Dominant action: "Move B from peg 0 to peg 2"

Frequency: ~50% of all predictions



Model shows SOME responsiveness to input but heavily biased.

```



\*\*Cause:\*\*

\- Gating mechanism helps but insufficient capacity

\- Partial learning of input-output patterns

\- Falls back to dominant pattern when uncertain

\- Better than pure collapse but still inadequate



\*\*Detection:\*\*

\- Measure action distribution entropy

\- Compare to expected uniform distribution

\- Check if any action appears >30% of time



\*\*Frequency:\*\* 50% for LSTM, 0% for other models



---



\### 3. STOP Bias



\*\*Observed in:\*\* GNN (42% of predictions vs 10% expected)



\*\*Symptom:\*\* Model over-predicts termination action.



\*\*Example:\*\*

```

State: \[\['A'], \[], \[]]

Goal: \[\[], \['A'], \[]]

Prediction: STOP (incorrect - puzzle not solved)



State: \[\[], \['B', 'C'], \[]]

Goal: \[\[], \[], \['B', 'C']]

Prediction: STOP (incorrect - more moves needed)

```



\*\*Cause:\*\*

\- Class imbalance in training data (10% STOP, 90% moves)

\- Global pooling in GNN averages away specific signals

\- STOP is "safe" neutral prediction

\- Model learns STOP as default when uncertain



\*\*Detection:\*\*

\- Compare STOP frequency to training distribution

\- Measure early termination rate (predict STOP before goal)

\- Check if STOP bias increases with puzzle complexity



\*\*Frequency:\*\* 42% vs 10% expected for GNN, not observed in other models



\*\*Positive Note:\*\* GNN shows most diverse predictions among small models. Shows 30+ different action types, unlike RNN/LSTM mode collapse.



---



\### 4. Oscillation Loops



\*\*Observed in:\*\* GPT-2 on OOD puzzles (80% of OOD failures)



\*\*Symptom:\*\* Model cycles between 2-3 states without making progress.



\*\*Example:\*\*

```

Step 1: \[\['A', 'B'], \[], \['C']]

Action: Move B from peg 0 to peg 1



Step 2: \[\['A'], \['B'], \['C']]

Action: Move B from peg 1 to peg 0



Step 3: \[\['A', 'B'], \[], \['C']]  (back to step 1)

Action: Move B from peg 0 to peg 1



Step 4: \[\['A'], \['B'], \['C']]  (cycle repeats indefinitely)

```



\*\*Cause:\*\*

\- Myopic reasoning: looks ahead 1 step, not full solution

\- No global planning mechanism

\- Each individual action appears locally reasonable

\- Lack of memory about visited states

\- More common on OOD because larger problems require longer planning horizon



\*\*Detection:\*\*

\- Track state history

\- Check if any state appears 3+ times

\- Measure cycle length: minimum steps between state repetitions

\- Compare recent states to state history



\*\*Frequency:\*\* 80% of GPT-2 OOD failures, 0% on ID puzzles (interesting - only emerges on harder problems)



\*\*Why GPT-2 specifically:\*\*

\- Decoder-only architecture

\- No separate encoding phase for global state understanding

\- Generates actions autoregressively without explicit planning representation



---



\### 5. Invalid Actions



\*\*Observed in:\*\* All small models (varying frequencies)



\*\*Symptom:\*\* Model predicts actions that violate game rules.



\*\*Examples:\*\*

```

Error Type 1: Wrong block

&nbsp;   State: \[\['A'], \['B'], \[]]

&nbsp;   Prediction: Move C from peg 0 to peg 2

&nbsp;   Problem: C is not on peg 0



Error Type 2: Empty source

&nbsp;   State: \[\['A'], \[], \['B']]

&nbsp;   Prediction: Move C from peg 1 to peg 0

&nbsp;   Problem: Peg 1 is empty



Error Type 3: Block not on top

&nbsp;   State: \[\['A', 'B'], \[], \[]]

&nbsp;   Prediction: Move A from peg 0 to peg 1

&nbsp;   Problem: A is not on top (B is on top)



Error Type 4: Invalid peg index

&nbsp;   Prediction: Move A from peg 3 to peg 1

&nbsp;   Problem: Only 3 pegs (0, 1, 2)

```



\*\*Cause:\*\*

\- Insufficient learning of game constraints

\- Pattern matching without understanding rules

\- May have learned "typical" actions without legality checking

\- Output appears syntactically correct but semantically invalid



\*\*Detection:\*\*

\- Check source peg has blocks

\- Check specified block is on top of source peg

\- Check peg indices in valid range (0-2)

\- Check block exists in current state



\*\*Frequency:\*\*

\- GNN: 15-20% of predictions

\- LSTM: 10-15% of predictions

\- Custom Transformer: 20-25% of predictions



\*\*Interpretation:\*\* Models learn surface patterns without understanding constraints.



---



\### 6. Premature Termination



\*\*Observed in:\*\* GNN primarily



\*\*Symptom:\*\* Model predicts STOP before goal is reached.



\*\*Example:\*\*

```

Current: \[\['A'], \['B'], \[]]

Goal: \[\[], \['A', 'B'], \[]]

Prediction: STOP

Problem: Goal not reached - A needs to move to peg 1

```



\*\*Cause:\*\*

\- Related to STOP bias

\- Model may not fully understand goal representation

\- Or lacks confidence in continuing

\- Defaults to termination when uncertain



\*\*Detection:\*\*

\- Check if current state matches goal state

\- If not equal and prediction is STOP: premature termination



\*\*Frequency:\*\* 25-30% of GNN failures



---



\## The "Confident False Negative" in Detail



\### What Makes It "Confident"



Models assign high probability to incorrect predictions:

```

Example (GNN):

&nbsp;   State: \[\[], \['A'], \[]]

&nbsp;   True Action: Move A to peg 0

&nbsp;   

&nbsp;   Model Predictions:

&nbsp;       STOP: 85% probability (incorrect)

&nbsp;       Move A to peg 0: 5% probability (correct)

&nbsp;       Other: 10% probability

&nbsp;   

&nbsp;   Model is CONFIDENT in wrong answer.

```



\### Why External Observers Cannot Detect It



\*\*Human evaluator examining trajectory:\*\*

```

Step 1: \[\[], \['A', 'B'], \[]]

&nbsp;   Model: "Move B to peg 2"

&nbsp;   Evaluator: "Seems reasonable - moving top block"



Step 2: \[\[], \['A'], \['B']]

&nbsp;   Model: "Move A to peg 0"

&nbsp;   Evaluator: "Makes sense - clearing middle peg"



Step 3: \[\['A'], \[], \['B']]

&nbsp;   Model: "STOP"

&nbsp;   Evaluator: "Looks solved - blocks are separated"



Actual Goal: \[\[], \[], \['A', 'B']]

Result: FAILED



Problem: Each step appeared logical, but internal goal understanding was wrong.

```



\*\*Why evaluator missed it:\*\*

\- Each individual action was legal

\- Each action seemed to make local progress

\- No obvious errors in action selection

\- Only final comparison reveals failure



\*\*The danger:\*\* In real deployment (education, healthcare), similar failures would go undetected until harm occurs.



---



\## Failure Patterns by Model Size



\### Small Models (<2M params)

```

Primary failure: Cannot learn task adequately

&nbsp;   - Mode collapse (RNN: 100%, LSTM: 50%)

&nbsp;   - STOP bias (GNN: 42%)

&nbsp;   - Invalid actions (15-25%)



Secondary failure: Distribution shift

&nbsp;   - Even when predictions seem reasonable

&nbsp;   - Cannot chain predictions correctly

&nbsp;   - Errors compound over steps

```



\### Large Models (>200M params, no pre-training)

```

Primary failure: Same as small models

&nbsp;   T5 from scratch behaves like Custom Transformer



Conclusion: Size alone insufficient

```



\### Large Pre-trained Models

```

Minimal failures on ID:

&nbsp;   - 1-3% failures

&nbsp;   - Usually on very complex cases

&nbsp;   - No mode collapse or bias issues



OOD failures increase:

&nbsp;   - GPT-2: 50% failure (oscillation dominant)

&nbsp;   - T5: 13-20% failure (more robust)

```



---



\## Detection Strategies



\### During Training



\*\*Monitor action distribution:\*\*

```

Expected: Roughly uniform across 30 move actions + 1 STOP

Actual (Elman RNN): 95% one action, 5% others → MODE COLLAPSE

Actual (GNN): 42% STOP, 58% others → STOP BIAS

```



\*\*Track prediction diversity:\*\*

```

Entropy = -sum(p\_i \* log(p\_i))

High entropy: Diverse predictions

Low entropy: Mode collapse

```



\### During Inference



\*\*State history tracking:\*\*

```

If state appears 3+ times: OSCILLATION

```



\*\*Action validation:\*\*

```

For each predicted action:

&nbsp;   - Check source peg not empty

&nbsp;   - Check block on top of source peg

&nbsp;   - Check peg indices valid

&nbsp;   - Check block exists

```



\*\*Goal checking:\*\*

```

If STOP predicted:

&nbsp;   - Verify current state == goal state

&nbsp;   - If not: PREMATURE TERMINATION

```



---



\## Why Standard Evaluation Misses These Problems



\### Teacher Forcing Evaluation

```

Process:

1\. Load (optimal\_state, optimal\_action) pair

2\. Model predicts action given optimal\_state

3\. Compare prediction to optimal\_action

4\. Report accuracy



Problem: Only tests single steps on OPTIMAL states

Missing: What happens when model handles its OWN predictions?

```



\### Why This Fails

```

Model learns: optimal\_state → optimal\_action mapping

Model faces: imperfect\_state → ??? (never saw this in training)

Result: Distribution shift causes failure



Analogy: Training driver on empty roads, testing on rush hour traffic

```



---



\## Implications for Deployment



\### For Educational AI



\*\*Scenario:\*\* AI tutor helping child with math problem

```

Step 1: "Let's isolate the variable"

&nbsp;   Evaluator: Correct pedagogical approach

&nbsp;   Student: Follows instruction



Step 2: "Subtract 3 from both sides"

&nbsp;   Evaluator: Correct algebraic step

&nbsp;   Student: Performs operation



Step 3: "Divide by 2"

&nbsp;   Evaluator: Correct simplification

&nbsp;   Student: Completes step



Result: Wrong answer



Problem: Internal concept of "isolating variable" was flawed from step 1,

but each individual instruction sounded reasonable. Child learned incorrect

mental model.

```



\### For Medical AI



\*\*Scenario:\*\* AI assisting physician with diagnosis

```

Step 1: "Patient presents with fever"

&nbsp;   Evaluator: Correct observation

&nbsp;   Physician: Notes fever



Step 2: "Fever suggests infection"

&nbsp;   Evaluator: Reasonable inference

&nbsp;   Physician: Considers infection



Step 3: "Prescribe broad-spectrum antibiotics"

&nbsp;   Evaluator: Follows from infection diagnosis

&nbsp;   Physician: Writes prescription



Result: Missed early-stage cancer



Problem: Internal weighting of symptoms emphasized fever, downweighted

other subtle indicators. Each reasoning step seemed medically sound in

isolation, but overall diagnostic reasoning was flawed.

```



---



\## What We Need for Safe Deployment



\### 1. Mechanistic Interpretability

\- Understand what representations model learns internally

\- Verify internal reasoning, not just outputs

\- Detect when internal states are malformed even if outputs seem reasonable



\### 2. Scalable Oversight

\- Verify multi-step reasoning chains we cannot manually check

\- Techniques: debate, process supervision, recursive reward modeling

\- Cannot rely on output inspection alone



\### 3. Adversarial Testing

\- Systematically search for failure modes

\- Test distribution shift scenarios

\- Evaluate on out-of-distribution inputs



\### 4. Process Supervision

\- Evaluate reasoning quality at each step

\- Not just final answer correctness

\- Reward correct reasoning paths, penalize shortcuts



---



\## Conclusion



The "confident false negative" problem demonstrates that:



1\. \*\*Standard evaluation is insufficient\*\* - high training accuracy does not predict deployment reliability

2\. \*\*Output inspection is insufficient\*\* - plausible-looking outputs can be internally flawed

3\. \*\*Scale alone is insufficient\*\* - need pre-training for robust representations

4\. \*\*Black boxes are dangerous\*\* - cannot verify correctness without mechanistic understanding



\*\*For AI safety:\*\* This research provides concrete evidence that we need mechanistic interpretability and scalable oversight before deploying AI systems in high-stakes applications. The gap between "appears to work" and "actually works safely" is real and measurable.

