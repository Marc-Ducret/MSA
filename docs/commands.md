# Useful Commands

## Minecraft Commands

A more complete guide to commands can be found on the [Minecraft Wiki](https://minecraft.gamepedia.com/Commands). Keep in mind that we are using Minecraft 1.12 and some information in the wiki might concern more recent versions.

### In the server console

* `op <username>` makes `<username>` an operator.
* `time set 0` sets daytime.
* `weather clear` removes the rain.
* `gamemode <survival|creative|adventure|spectator> <username>` sets `<username>`'s gamemode
* `tp <username> <x> <y> <z>` teleports `<username>` to `(<x>, <y>, <z>)`.
* `stop` stops the server.

### From an operator

Commands must be typed in the chat and preceded by `/`. All server commands can be used by an operator.

* `tp <x> <y> <z>` teleports the operator to `(<x>, <y>, <z>)`.
* `gamemode <survival|creative|adventure|spectator>` sets the operator's gamemode.

## Minecraft Server Agents Commands

* `tps` displays the number of *ticks* processed per second.
* `ft` controls the execution speedup feature:
    * `ft start` enables *fast ticking*.
    * `ft <duration>` enables *fast ticking* for `<duration>` ticks.
    * `ft stop` disables *fast ticking* (disable it before using `stop`, `compile` and maybe some other commands).
* `env` allows manual creation of environments.
    * `env add <env-id> <env-type>` creates an environment of type `<env-type>` with unique name `<env-id>`. `<env-type>` is the name of the environment's Java class without `Environment` and eventually followed by parameters: an example would be `Pattern[8,2,4]`.
    * `env remove <env-id>` removes environment `<env-id>`.
    * `env player <env-id> <username> ...` adds `<username>` (multiple players can be specified) to `<env-id>` as a human actor.
* `exec <class> <static-method>` executes `<static-method>` of `<class>`: an example would be `exec edu.usc.thevillagers.serversideagent.env.EnvironmentTrade writeStats`.
* `cst <name> <value>` set constant `<name>` to `<value>`.
    * `cst skip <int>` specifies the number of ticks to skip between environment updates (default: 0).
    * `cst speed <float>` specifies agents' movement speed factor (default: 1).
    * `cst prop <float>` specifies agents' discrete actions probability factor (default: 1).
    * `cst pitch <float>` specifies agents' maximum pitch magnitude (default: 90).
* `rec` controls the recording features.
    * `rec start <from-x> <from-y> <from-z> <to-x> <to-y> <to-z>` starts recording of the world in the specified box. No other recording can procced at the same time.
    * `rec stop` stops the current recording and saves it to `forge/mod/run/tmp/records`
* `exp <env-type> <episodes> <x> <y> <z> <username> ...` starts an experiment with specified human subjects within an environment of `<env-type>` at the specified position and records it for `<episodes>`. Only one experiment or recording can happen at one time. Experiments can be stopped early by `rec stop` and `env remove exp`.
* `compile <record> <env-type>` compiles the only recording containing `<record>` into the *observation-action* dataset `forge/mod/run/tmp/imitation/<record>.h5` using the observation and actions spaces defined by `<env-type>`.
