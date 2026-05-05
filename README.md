# Procedural world generation

A 2D sandbox game with procedural world generation, particle physics simulation, and dynamic lighting system built in Python.

## Features

- **Procedural World Generation** – Infinite explorable worlds with varied terrain generation
- **Particle Physics** – Advanced particle simulation with gravity and collision detection
- **Dynamic Lighting System** – Real-time light rendering and shadow casting
- **Interactive Gameplay** – Mine blocks, destroy matter, and manipulate the environment
- **GPU-Accelerated Computation** – OpenCL kernels for optimized physics calculations
- **Chunked Rendering** – Efficient viewport culling and chunk-based world updates

## Technology Stack

- **Python 3** – Core game logic
- **Pygame** – Graphics rendering and event handling
- **NumPy** – Mathematical operations and array optimization
- **OpenCL** – GPU-accelerated diffusion and particle physics
- **PIL** – Sprite and collision map loading

## Project Structure

```
├── main.py                 # Entry point and game loop
├── game_context.py         # Core game engine and rendering pipeline
├── game_content.py         # Game objects, physics, and world data
├── diffuzer.py            # Canvas system and physics simulation
├── kernels.cl             # OpenCL kernels for GPU computation
├── sprites/               # Character and entity textures
├── backgrounds/           # Tileable background images
└── collisions/            # Collision geometry maps
```

## Installation

### Prerequisites
- Python 3.8+
- OpenCL support (GPU or CPU driver required)

### Setup

```bash
pip install pygame numpy pillow pyopencl
```

## Running the Game

```bash
python main.py
```

### Controls
- **E** – Destroy terrain at cursor position
- **Q** – Place sand blocks
- **Mouse** – Aim and interact with the world
- **ESC** – Exit game

## How It Works

### World System
The game uses a chunked world divided into manageable sections for efficient updates. The `diffuzer` module simulates particle interactions using OpenCL, computing physics on the GPU for performance.

### Lighting
Dynamic lighting is calculated per frame, affecting all rendered elements. Light sources from game objects interact with the particle system in real-time.

### Physics
- Gravity affects all particles uniformly (0.4 acceleration)
- Air friction reduces velocity (0.99 damping factor)
- Particle-to-particle collisions use sprite-based collision maps
- Custom matter types (sand, stone, sandstone) with unique properties

## Performance Notes

- World size: 3000×3000 pixels
- Canvas rendering: 1000×1000 pixels
- Screen resolution: 1920×1080 (fullscreen)

## Future Enhancements

- Network multiplayer support
- Additional terrain types and biomes
- Advanced weather system
- Crafting and building mechanics
- Save/load world persistence
