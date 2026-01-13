\# T5 Fine-tuning for Hierarchical Reasoning



\## Overview



T5 (Text-to-Text Transfer Transformer) is a pre-trained encoder-decoder model from Google. We fine-tuned T5-base (223M parameters) on Block World puzzles to test if large pre-trained models overcome the "Illusion of Thinking" that affects small models.



\*\*Key Finding:\*\* Pre-training is essential. T5 with pre-trained weights achieves 99%+ accuracy, while T5 from scratch (same architecture, same size) achieves only 4% - identical to small models.



---



\## Two Experiments



\### Experiment 1: T5 from Scratch (Random Initialization)

```

Model: T5-base architecture (223M params)

Initialization: Random weights (no pre-training)

Training: 20 epochs, learning rate 3e-4



Results:

&nbsp;   Teacher Forcing: ~4%

&nbsp;   Autoregressive ID: ~0.5%

&nbsp;   Autoregressive OOD: 0%



Conclusion: Architecture and scale alone INSUFFICIENT

Behaves identically to Custom Transformer (0.8M params, 4.01% TF)

```



\### Experiment 2: T5 with Pre-trained Weights (Fine-tuning)

```

Model: T5-base (223M params)

Initialization: Pre-trained weights from HuggingFace

Training: 3 epochs, learning rate 5e-5



Results (Symbolic):

&nbsp;   Teacher Forcing: 97.87%

&nbsp;   Autoregressive ID: 97.45%

&nbsp;   Autoregressive OOD: 80.33%



Results (Natural Language):

&nbsp;   Teacher Forcing: 99.26%

&nbsp;   Autoregressive ID: 99.09%

&nbsp;   Autoregressive OOD: 85.67%



Conclusion: Pre-training provides robust representations

Natural language better than symbolic (leverages pre-training)

```



---



\## Data Format



\### Symbolic Representation

```

Input (State):

&nbsp;   "<SOS> PEG\_A #A #B <SEP> PEG\_B <BLANK> <SEP> PEG\_C #C <SEP> <EOS>"



Target (Action):

&nbsp;   "D #A N1 N3"  (Move A from peg 0 to peg 2)

```



\### Natural Language Representation

```

Input (State):

&nbsp;   "Peg 0: A, B. Peg 1: empty. Peg 2: C."



Target (Action):

&nbsp;   "Move A from peg 0 to peg 2"

```



---



\## Fine-tuning Procedure



\### Setup

```

LOAD T5-base model from HuggingFace

LOAD T5 tokenizer



SET device = GPU if available



FOR representation in \[Symbolic, Natural]:

&nbsp;   

&nbsp;   # Prepare dataset

&nbsp;   CREATE training examples:

&nbsp;       FOR each (state, action) pair:

&nbsp;           input\_text = encode\_state(state, representation)

&nbsp;           target\_text = encode\_action(action, representation)

&nbsp;           

&nbsp;           input\_ids = tokenizer(input\_text)

&nbsp;           target\_ids = tokenizer(target\_text)

&nbsp;           

&nbsp;           STORE (input\_ids, target\_ids)

&nbsp;   

&nbsp;   # Training configuration

&nbsp;   SET batch\_size = 8 (smaller due to large model)

&nbsp;   SET learning\_rate = 5e-5 (lower than from-scratch)

&nbsp;   SET num\_epochs = 3 (fewer due to pre-training)

&nbsp;   SET optimizer = AdamW

&nbsp;   

&nbsp;   # Fine-tuning loop

&nbsp;   FOR epoch in \[1..3]:

&nbsp;       

&nbsp;       FOR batch in training\_data:

&nbsp;           

&nbsp;           # Forward pass

&nbsp;           outputs = model(

&nbsp;               input\_ids=batch.input\_ids,

&nbsp;               labels=batch.target\_ids

&nbsp;           )

&nbsp;           

&nbsp;           loss = outputs.loss

&nbsp;           

&nbsp;           # Backward pass

&nbsp;           loss.backward()

&nbsp;           optimizer.step()

&nbsp;           optimizer.zero\_grad()

&nbsp;       

&nbsp;       # Evaluate

&nbsp;       tf\_acc = evaluate\_teacher\_forcing(model, val\_data)

&nbsp;       auto\_acc = evaluate\_autoregressive(model, val\_puzzles)

&nbsp;       

&nbsp;       PRINT epoch, tf\_acc, auto\_acc

&nbsp;   

&nbsp;   # Final evaluation

&nbsp;   EVALUATE on ID (N=1-7) and OOD (N=8-10)

```



---



\## Autoregressive Generation with T5



\### Teacher Forcing Evaluation

```

FOR each (state, action) pair in validation:

&nbsp;   

&nbsp;   input\_ids = tokenize(state\_description)

&nbsp;   target\_ids = tokenize(action\_description)

&nbsp;   

&nbsp;   # Model predicts given correct state

&nbsp;   predictions = model.generate(

&nbsp;       input\_ids=input\_ids,

&nbsp;       max\_length=action\_length

&nbsp;   )

&nbsp;   

&nbsp;   IF predictions == target\_ids:

&nbsp;       correct += 1



accuracy = correct / total

```



\### Autoregressive Puzzle Solving

```

FOR each puzzle in test\_set:

&nbsp;   

&nbsp;   current\_state = puzzle.start\_state

&nbsp;   goal\_state = puzzle.goal\_state

&nbsp;   steps = 0

&nbsp;   

&nbsp;   WHILE current\_state != goal\_state AND steps < max\_steps:

&nbsp;       

&nbsp;       # Encode current state (may be imperfect due to previous predictions)

&nbsp;       state\_text = encode\_state(current\_state, representation)

&nbsp;       input\_ids = tokenize(state\_text)

&nbsp;       

&nbsp;       # Generate action

&nbsp;       output\_ids = model.generate(

&nbsp;           input\_ids=input\_ids,

&nbsp;           max\_length=20

&nbsp;       )

&nbsp;       

&nbsp;       action\_text = detokenize(output\_ids)

&nbsp;       action = parse\_action(action\_text)

&nbsp;       

&nbsp;       # Validate and apply action

&nbsp;       IF not is\_valid\_action(action, current\_state):

&nbsp;           RECORD failure (invalid action)

&nbsp;           BREAK

&nbsp;       

&nbsp;       current\_state = apply\_action(current\_state, action)

&nbsp;       steps += 1

&nbsp;   

&nbsp;   IF current\_state == goal\_state:

&nbsp;       RECORD success

&nbsp;   ELSE:

&nbsp;       RECORD failure

```



---



\## Key Differences: From-Scratch vs Fine-tuning



\### From Scratch (Random Initialization)

```

Weights: Random initialization

Training: Must learn everything from Block World data

Epochs: 20 (slow convergence)

Learning rate: 3e-4 (standard)



Challenges:

&nbsp;   - No pre-trained representations

&nbsp;   - Limited task-specific data (3,763 samples)

&nbsp;   - Must learn language understanding from scratch

&nbsp;   - Fails to generalize



Result: ~4% TF, ~0.5% Auto (catastrophic failure)

```



\### Fine-tuning (Pre-trained Weights)

```

Weights: Pre-trained on massive text corpus

Training: Adapt existing knowledge to Block World

Epochs: 3 (fast convergence)

Learning rate: 5e-5 (smaller, preserve pre-training)



Advantages:

&nbsp;   - Rich language representations

&nbsp;   - Understanding of compositional structure

&nbsp;   - Robust to distribution shift

&nbsp;   - Generalizes well



Result: 99% TF, 99% Auto (success)

```



---



\## Why Natural Language Works Better for T5



\### Symbolic Encoding

```

Format: "<SOS> PEG\_A #A <SEP>..."



T5 Performance:

&nbsp;   Teacher Forcing: 97.87%

&nbsp;   Autoregressive ID: 97.45%

&nbsp;   OOD: 80.33%



Issue: Synthetic tokens (#A, PEG\_A) not in pre-training

Model must learn these token meanings from scratch

```



\### Natural Language Encoding

```

Format: "Peg 0: A, B. Peg 1: empty. Peg 2: C."



T5 Performance:

&nbsp;   Teacher Forcing: 99.26%

&nbsp;   Autoregressive ID: 99.09%

&nbsp;   OOD: 85.67%



Advantage: Natural English aligns with pre-training

Model leverages existing language understanding

Smaller gap means better generalization

```



\*\*Insight:\*\* Representation should match pre-training distribution for maximum benefit.



---



\## Comparison: All T5 Variants

```

Model                   Init        TF Acc    Auto ID    Auto OOD

------------------------------------------------------------------

T5 from scratch         Random      ~4%       ~0.5%      0%

T5 Symbolic             Pre-trained 97.87%    97.45%     80.33%

T5 Natural              Pre-trained 99.26%    99.09%     85.67%



Conclusion:

&nbsp;   1. Pre-training is ESSENTIAL (4% → 99%)

&nbsp;   2. Scale without pre-training = failure

&nbsp;   3. Natural language > Symbolic for pre-trained models

&nbsp;   4. Encoder-decoder > Decoder-only (compare GPT-2: 49.67% OOD)

```



---



\## Critical Insight for AI Safety



\*\*Pre-training provides robust representations that generalize beyond training distribution.\*\*



Without pre-training:

\- 223M param model performs like 0.8M param model

\- Cannot handle distribution shift

\- Fails catastrophically in deployment



With pre-training:

\- Minimal gap between training and deployment (0.17pp)

\- Handles imperfect states robustly

\- Generalizes to larger, unseen problems



\*\*Implication:\*\* For safe deployment in high-stakes applications (education, healthcare), we need models with rich pre-trained representations that can handle distribution shift, not just large parameter counts.

