package edu.usc.thevillagers.serversideagent.env.actuator;

import edu.usc.thevillagers.serversideagent.agent.Actor;

/**
 * Actuator of dimension 2 that enables horizontal movement.
 */
public class ActuatorForwardStrafe extends Actuator {

	public ActuatorForwardStrafe() {
		super(2);
	}

	@Override
	public void act(Actor actor) {
		actor.actionState.forward = values[0];
		actor.actionState.strafe = values[1];
	}
}
