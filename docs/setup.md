 # Setup

 ## Directory structure

 Create a folder named `forge` and a folder `mod` inside. Then checkout this repository inside of `/forge/mod/src`. Confirm that `src` contains `setup.sh`.

 ## Dependencies

 * Java Development Kit 8
 * Python 3
    * Tensorflow
    * Numpy
    * OpenAI's baselines and gym
    * PyTorch
    * probably more (TODO: which ones?)
 * (UNIX only) unzip command

## Running the setup script

To avoid issues, make sure that `forge` only contains `mod` only containing `src`.

### Unix

Open a terminal in `forge/mod/src/` and execute `./setup.sh`.

### Windows

Open a `PowerShell` in `forge\mod\src\` and execute `.\setup.ps1`.

## Starting the Minecraft Server

Run `./gradlew runServer -Pusername=` in `forge/mod/`.

## Starting a Minecraft Client

Run `./gradlew runClient -Pusername=<name>` in `forge/mod/` having replaced `<name>` with your desired username.

## Running agent scripts

Scripts should be placed in `forge/mod/src/python/`. They should be run from the `forge/mod/run` folder using the symbolic link to the `python` folder.

An example would be: running `python python/agent_dummy.py Pattern` in `forge/mod/run/`
