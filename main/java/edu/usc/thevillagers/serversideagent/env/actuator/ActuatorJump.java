package edu.usc.thevillagers.serversideagent.env.actuator;

import edu.usc.thevillagers.serversideagent.agent.Actor;

/**
 * Actuator of dimension 1 that enables jumping.
 */
public class ActuatorJump extends Actuator {

	public ActuatorJump() {
		super(1);
	}

	@Override
	public void act(Actor actor) {
		actor.actionState.jump = values[0] > .5F;
	}
}
