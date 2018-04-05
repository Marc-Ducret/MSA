# Minecraft Forge

This document aims at explaining why [Minecraft Forge](https://files.minecraftforge.net/) is useful and how it should be used.

## The need for such a tool

First of all, Minecraft's sources have never been released by Mojang. Fortunately, Java is a rather friendly language to decompile: names are preserved and most of the code can be recompiled without modification. However, Mojang purposely obfuscate their jars before release: all names (class, variables, functions) become abstract sequences of letters (aa.class, ab.class, ac.class, ...). Moreover, the obfuscated names change at each Minecraft update. [MCP (Mod Coder Pack)](https://minecraft.gamepedia.com/Programs_and_editors/Mod_Coder_Pack) is a community made tool that decompiles Minecraft, translates the names to be readable and fixes the decompilation errors.

Then, if each mod was to incorporate itself by altering Minecraft's code, combining mods would require complex merging of classes that seem very hard to automate. On the other hand, Minecraft Forge provides a rich API the enable the creation of mods that are loaded by Forge and rely on events to function.

While the diversity of Forge events already enables a lot of complex behaviours using events will always be limited in a sense or another. However, Forge provides tools for runtime altering of Minecraft's classes. Doing so isn't easy even for simple modifications and compatibility between mods isn't assured but it can be sometimes necessary.

## Setup

The setup of Minecraft Forge is fairly straight forward and well explained [here](http://mcforge.readthedocs.io/en/latest/gettingstarted/).

## Developing with Forge

The global ideas and concepts of Forge are explained in their [official documentation](http://mcforge.readthedocs.io/en/latest/). Details about implementation can be learned by reading Forge's code (you should be able to access it after a successful setup). Another option is to look through open source mods, there are many of those on GitHub.
