# Minecraft Mechanics

This document will outline how Minecraft runs and simulates its world.
It will refer to various articles on the [official Minecraft Wiki](https://minecraft.gamepedia.com) for more details. This wiki not only aims at new players but also at the significant technical community of Minecraft. Therefore, it provides accurate information about the gameplay but also about how the game engine works.

## Objects Representations

In a Minecraft world, objects have different representations to match their dynamism and complexity.
* [Blocks](https://minecraft.gamepedia.com/Block) are simple, slowly evolving and very light computationally
* [Entities](https://minecraft.gamepedia.com/Entity) are complex, constantly evolving and more heavy
* [Block Entities](https://minecraft.gamepedia.com/Block_entity) (AKA Tile Entities) are a hybrid approach to allow
 for more complex blocks

## Space

A Minecraft world is pseudo infinite (limited by integer and floating point limitations more or less). Therefore, loading is not trivial. The world is organized in [chunks](https://minecraft.gamepedia.com/Chunk), some of which are loaded at some point in time. A good approximation is to consider that chunks are loaded when they are close enough to a player (the distance is a parameter of the server).

A single Minecraft server contains multiple worlds or [dimensions](https://minecraft.gamepedia.com/Chunk).

## Time

The simulation evolves with constant time steps of 50 ms. Each of those updates is called a
['tick'](https://minecraft.gamepedia.com/Chunk).

## Network Architecture

Minecraft is a multiplayer game and therefore has its [server](https://minecraft.gamepedia.com/Server) and [clients](https://minecraft.gamepedia.com/Minecraft_launcher). The client acts as an interface between the player and the world. The approach is a relaxed authoritarian server in the sense that most things are totally controlled by the server while clients follow (and anticipate to some extend) their evolution but player movement and interactions are mostly decided by the player's client and only checked by the server to prevent abuse.

In this setting, there is no strong synchronization between clients and server (in contrast with [Deterministic Lockstep](https://gafferongames.com/post/deterministic_lockstep/) approaches).
