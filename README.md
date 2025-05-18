# AlphaZero STRANDS Integration

A powerful integration of AlphaZero's game-playing capabilities with AWS STRANDS Agents for enhanced reasoning, memory persistence, and advanced decision-making.

## Overview

This project combines two powerful AI frameworks:
- **AlphaZero**: A reinforcement learning algorithm that masters games through self-play and deep neural networks
- **AWS STRANDS Agents**: A model-driven framework for building flexible, capable AI agents

By integrating these technologies, we create a more robust game-playing agent with:
- LLM-enhanced position evaluation
- Persistent memory across games
- Dynamic tool selection during gameplay
- Improved strategic reasoning

## Features

- **Model-driven decision augmentation**: Uses STRANDS' model-driven approach to enhance AlphaZero's decision-making
- **Hybrid control flow**: STRANDS' agentic loop controls high-level strategy while AlphaZero's MCTS handles tactical decisions
- **Memory persistence**: Sophisticated memory management for learning across sessions
- **Multi-agent game analysis**: Specialized agents for different aspects of game analysis
- **Progressive learning**: Start with faster models and graduate to more capable ones

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                STRANDS Agent Container                  │
│  ┌─────────────┐   ┌────────────────┐   ┌───────────┐  │
│  │ LLM (Claude,│   │    Agentic     │   │  Memory   │  │
│  │ Llama, etc) │◄─►│      Loop      │◄─►│  Manager  │  │
│  └─────────────┘   └────────────────┘   └───────────┘  │
│           │               │                   │        │
│           ▼               ▼                   ▼        │
│  ┌─────────────┐   ┌────────────────┐   ┌───────────┐  │
│  │    Tools    │   │ AlphaZero MCTS │   │  State    │  │
│  │  Ecosystem  │◄─►│     Tool       │◄─►│  Manager  │  │
│  └─────────────┘   └────────────────┘   └───────────┘  │
└────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/JohnJBoren/alphazero-strands-integration.git
cd alphazero-strands-integration

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.8+
- Ollama (locally installed)
- AWS STRANDS Agents SDK
- NumPy
- Matplotlib (for visualization)

## Quick Start

```python
from alphazero_strands import STRANDSAlphaZero
from games import Chess

# Initialize the integrated agent
agent = STRANDSAlphaZero(
    game_class=Chess,
    model_path="models/chess_model.pt",  # Optional: load existing model
)

# Train through self-play
agent.train(num_games=100)

# Play against the agent
agent.play_against_human()
```

## Core Components

- **games/** - Implementations of various games (Chess, Connect4, etc.)
- **mcts/** - Enhanced Monte Carlo Tree Search with LLM integration
- **neural_net/** - Neural network architecture for policy and value prediction
- **memory/** - Advanced memory management with STRANDS integration
- **tools/** - Tools for game analysis, visualization, and deployment
- **agents/** - Multi-agent implementation for collaborative analysis

## Examples

See the `examples/` directory for complete usage examples:

- `examples/chess_training.py` - Train an agent to play chess
- `examples/connect4_visualization.py` - Visualize MCTS search trees for Connect4
- `examples/multi_agent_analysis.py` - Analyze games with specialized agents

## License

MIT License

## Acknowledgments

- AWS STRANDS Agents team for their excellent framework
- AlphaZero paper and implementations
- The Ollama project for making local LLM usage accessible
