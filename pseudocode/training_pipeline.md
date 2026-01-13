\# Training Pipeline for Hierarchical Reasoning Models



\## Overview



Single-step supervised learning on optimal action trajectories.



\*\*Training Objective:\*\* Learn to predict next optimal action given current state.



\*\*Key Limitation:\*\* Models trained on optimal states cannot handle distribution shift to their own (imperfect) predictions during deployment.



---



\## Data Preparation



\### Dataset Structure

```

Training: N=1-7 blocks (3,763 samples from 549 puzzles)

Testing: N=8-10 blocks (2,913 samples from 300 puzzles)



Each sample: (current\_state, optimal\_action, goal\_state)

```



\### State Encoding

```

FOR each architecture:

&nbsp;   IF RNN/LSTM/Transformer:

&nbsp;       Encode state as token sequence

&nbsp;   ELSE IF GNN:

&nbsp;       Encode state as graph structure

&nbsp;   ELSE IF T5:

&nbsp;       Encode state as natural language text

```



\### Action Encoding

```

Action space: 31 classes

&nbsp;   - 30 move actions (10 blocks × 3 destination pegs)

&nbsp;   - 1 STOP action

```



---



\## Training Loop

```

INITIALIZE model with architecture-specific parameters

INITIALIZE optimizer (AdamW, learning rate 3e-4)

INITIALIZE loss function (CrossEntropyLoss)



FOR each epoch in \[1..20]:

&nbsp;   

&nbsp;   SET model to training mode

&nbsp;   

&nbsp;   FOR each batch in training\_data:

&nbsp;       

&nbsp;       1. Load (state, action) pairs

&nbsp;       

&nbsp;       2. Encode states:

&nbsp;          state\_input = encode(state, architecture)

&nbsp;       

&nbsp;       3. Forward pass:

&nbsp;          predictions = model(state\_input)

&nbsp;       

&nbsp;       4. Compute loss:

&nbsp;          loss = CrossEntropyLoss(predictions, action\_labels)

&nbsp;       

&nbsp;       5. Backward pass:

&nbsp;          loss.backward()

&nbsp;          optimizer.step()

&nbsp;          optimizer.zero\_grad()

&nbsp;   

&nbsp;   END FOR batch

&nbsp;   

&nbsp;   # Evaluation

&nbsp;   teacher\_forcing\_acc = evaluate\_single\_step(model, validation\_data)

&nbsp;   autoregressive\_acc = evaluate\_multi\_step(model, validation\_puzzles)

&nbsp;   

&nbsp;   PRINT "Epoch", epoch, "TF:", teacher\_forcing\_acc, "Auto:", autoregressive\_acc

&nbsp;   

&nbsp;   # Key observation: gap = teacher\_forcing\_acc - autoregressive\_acc

&nbsp;   # Large gap indicates "Illusion of Thinking"



END FOR epoch

```



---



\## Architecture-Specific Details



\### For RNN/LSTM (Encoder-Decoder)

```

Encoder:

&nbsp;   Input: Token sequence (state)

&nbsp;   Process: Recurrent layers encode sequence → hidden state

&nbsp;   Output: Context vector



Decoder:

&nbsp;   Input: Context vector + start token

&nbsp;   Process: Recurrent layers decode → action sequence

&nbsp;   Output: Action tokens



Training: Teacher forcing (feed correct previous tokens)

Inference: Autoregressive (feed model's own predictions)

```



\### For Transformer

```

Encoder:

&nbsp;   Input: Token sequence (state)

&nbsp;   Process: Self-attention layers

&nbsp;   Output: Contextualized representations



Decoder:

&nbsp;   Input: Encoder output + target tokens (shifted)

&nbsp;   Process: Masked self-attention + cross-attention

&nbsp;   Output: Action predictions



Training: Parallel decoding with masking

Inference: Autoregressive generation

```



\### For GNN

```

Graph Construction:

&nbsp;   Input: State \[blocks on pegs]

&nbsp;   Process: Create nodes (blocks + pegs), add edges (relationships)

&nbsp;   Output: Graph structure



Graph Processing:

&nbsp;   Input: Graph with node features

&nbsp;   Process: Message passing (nodes exchange information)

&nbsp;   Output: Updated node representations



Global Pooling:

&nbsp;   Input: All node representations

&nbsp;   Process: Aggregate (mean/max/attention)

&nbsp;   Output: Graph-level vector



Classification:

&nbsp;   Input: Graph vector

&nbsp;   Process: MLP layers

&nbsp;   Output: Action class (31 classes)

```



\### For T5 (Pre-trained)

```

Encoder:

&nbsp;   Input: Natural language state description

&nbsp;   Process: T5 encoder layers (pre-trained)

&nbsp;   Output: Contextualized representations



Decoder:

&nbsp;   Input: Encoder output + target action text (shifted)

&nbsp;   Process: T5 decoder layers (pre-trained)

&nbsp;   Output: Action text predictions



Fine-tuning:

&nbsp;   Freeze: None (full fine-tuning)

&nbsp;   Learning rate: Lower than from-scratch (5e-5)

&nbsp;   Epochs: Fewer (3 epochs sufficient due to pre-training)

```



---



\## Critical Insight: The Distribution Shift Problem



\### Training Distribution

```

Model sees: Optimal states from expert trajectories

State quality: Always correct, always on optimal path

Model learns: Pattern matching on optimal states

```



\### Deployment Distribution

```

Model sees: States resulting from its own predictions

State quality: May be suboptimal or incorrect

Model faces: Distribution shift - never saw these states in training

```



\### Result

```

Training accuracy: High (models learn patterns well)

Deployment accuracy: Low (models fail on shifted distribution)



This is the "Illusion of Thinking"

```



---



\## Experimental Results Pattern



\### Small Models (<2M params)

```

Training:

&nbsp;   Loss converges

&nbsp;   Accuracy improves

&nbsp;   Appears successful



Deployment:

&nbsp;   Catastrophic failure (<1.3% success)

&nbsp;   Mode collapse, oscillation, invalid actions

&nbsp;   Cannot handle own predictions



Conclusion: Architecture insufficient without scale

```



\### Large Pre-trained Models (>200M params)

```

Training:

&nbsp;   Loss converges quickly (3 epochs)

&nbsp;   Accuracy very high (>97%)

&nbsp;   

Deployment:

&nbsp;   Success (>97% on ID, >49% on OOD)

&nbsp;   Handles own predictions robustly

&nbsp;   Generalizes to larger problems



Conclusion: Pre-training provides robust representations

```



---



\## Why Single-Step Supervision Fails for Small Models



1\. \*\*Limited Capacity:\*\* Cannot learn robust state representations

2\. \*\*Distribution Shift:\*\* Training states ≠ deployment states

3\. \*\*No Correction Mechanism:\*\* Errors compound without recovery

4\. \*\*Pattern Matching:\*\* Learn shortcuts instead of reasoning rules



\*\*Solution:\*\* Scale + pre-training provides richer representations that generalize beyond training distribution.

