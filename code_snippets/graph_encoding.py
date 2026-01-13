"""
State Encoding Strategies for Different Architectures

Demonstrates how representation choice affects learning efficiency.
Key Finding: Optimal representation depends on architecture and scale.

Three approaches tested:
1. Symbolic tokens (RNN/LSTM/Transformer/GPT-2)
2. Natural language (T5)
3. Graph structure (GNN)
"""

# ============================================================================
# REPRESENTATION COMPARISON
# ============================================================================

"""
Example State: [['A', 'B'], [], ['C']]

SYMBOLIC ENCODING (for RNN/LSTM/Transformer):
Format: <SOS> PEG_A #A #B <SEP> PEG_B <BLANK> <SEP> PEG_C #C <SEP> <EOS>

Vocabulary: ~23 tokens
- Control: <PAD>, <SOS>, <EOS>, <SEP>
- Pegs: PEG_A, PEG_B, PEG_C
- Blocks: #A, #B, ..., #J
- Actions: D (displace), N1, N2, N3 (peg indices), STOP

Length: 10-30 tokens depending on state complexity
Used for: Custom Transformer, Elman RNN, LSTM, GPT-2

Results:
- Custom Transformer: 4.01% TF
- LSTM: 2.99% TF
- Elman RNN: 0.55% TF
- GPT-2: 99.47% TF (pre-trained)


NATURAL LANGUAGE ENCODING (for T5):
Format: "Peg 0: A, B. Peg 1: empty. Peg 2: C."

Action Format: "Move A from peg 0 to peg 1"

Length: 30-80 characters depending on state
Leverages: Pre-trained language understanding

Results:
- T5 Natural: 99.26% TF, 99.09% ID (BEST)
- T5 Symbolic: 97.87% TF, 97.45% ID

Insight: Natural language works better for pre-trained models
because it aligns with training distribution.


GRAPH ENCODING (for GNN):
Structure:
- Nodes: Blocks + Pegs
- Node features: [block_id, peg_location, height_on_peg]
- Edges: Relationships (on-top-of, on-peg)

Example for [['A', 'B'], [], ['C']]:
Nodes: 6 (3 blocks + 3 pegs)
Edges: 8 (relationships between blocks and pegs)

Compact: 6 nodes vs 13 tokens (symbolic)
Explicit: Relationships encoded in structure

Results:
- GNN: 10.85% TF with only 70K params (BEST small model)

Insight: Graph structure provides strong inductive bias
for hierarchical reasoning in small models.
"""


# ============================================================================
# KEY FINDINGS: REPRESENTATION MATTERS BUT SCALE MATTERS MORE
# ============================================================================

"""
Small Models (<2M params):
Graph > Symbolic
- GNN 10.85% vs Custom Transformer 4.01%
- Graph structure helps when limited capacity
- But all still fail autoregressively (<1.3%)

Large Pre-trained Models (>200M params):
Natural > Symbolic
- T5 Natural 99.09% vs T5 Symbolic 97.45%
- Pre-trained language understanding provides advantage
- Both succeed autoregressively (>97%)

Scale Comparison (same architecture, different scale):
T5 from scratch (223M, no pre-training): ~4% TF, ~0.5% Auto
T5 pre-trained (223M): 99.26% TF, 99.09% Auto

Conclusion:
Architecture and representation provide marginal improvements
(2-3x better performance in small models), but scale + pre-training
provides 100-300x improvement. You cannot replace scale with
clever representations.
"""


# ============================================================================
# ENCODING FUNCTIONS (CONCEPTUAL)
# ============================================================================

def encode_state_symbolic(state):
    """
    Convert state to token sequence.
    
    Process:
    1. Start with <SOS>
    2. For each peg:
        - Add peg identifier
        - Add blocks (or <BLANK> if empty)
        - Add <SEP>
    3. End with <EOS>
    
    Returns: List of token IDs
    """
    pass


def encode_state_natural_language(state):
    """
    Convert state to natural language description.
    
    Process:
    1. Describe each peg: "Peg N: [blocks]"
    2. Use "empty" for empty pegs
    3. Join with periods
    
    Returns: String
    """
    pass


def encode_state_graph(state):
    """
    Convert state to graph structure.
    
    Process:
    1. Create node for each block (features: id, peg, height)
    2. Create node for each peg (features: peg_id, index, -1)
    3. Add edges:
        - Block to peg it's on (bidirectional)
        - Block to block below (bidirectional)
    
    Returns: Graph data structure (nodes + edges)
    """
    pass


# ============================================================================
# VISUALIZATION: REPRESENTATION SIZE
# ============================================================================

"""
State Complexity vs Representation Size:

N=1 (1 block):
- Symbolic: ~8 tokens
- Natural: ~20 chars
- Graph: 4 nodes, 2 edges

N=5 (5 blocks):
- Symbolic: ~18 tokens
- Natural: ~50 chars
- Graph: 8 nodes, 14 edges

N=10 (10 blocks):
- Symbolic: ~33 tokens
- Natural: ~100 chars
- Graph: 13 nodes, 38 edges

Growth Rate:
- Symbolic: Linear in number of blocks
- Natural: Linear in number of blocks
- Graph: Quadratic in edges (but captures relationships explicitly)

For small models with limited capacity:
Graph's explicit structure compensates for quadratic growth,
resulting in better learning efficiency.
"""