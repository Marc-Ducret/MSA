package edu.usc.thevillagers.serversideagent.env.actuator;

import edu.usc.thevillagers.serversideagent.agent.Agent;

/**
 * Actuator of dimension 2 that enables horizontal movement.
 */
public class ActuatorForwardStrafe extends Actuator {

	public ActuatorForwardStrafe() {
		super(2);
	}

	@Override
	public void act(Agent agent) {
		agent.actionState.forward = values[0];
		agent.actionState.strafe = values[1];
	}
}
