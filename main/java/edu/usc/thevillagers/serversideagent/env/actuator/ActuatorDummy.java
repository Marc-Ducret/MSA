package edu.usc.thevillagers.serversideagent.env.actuator;

import edu.usc.thevillagers.serversideagent.agent.Actor;

/**
 * Actuator with no effect.
 */
public class ActuatorDummy extends Actuator {

	public ActuatorDummy(int dim) {
		super(dim);
	}

	@Override
	public void act(Actor actor) {
	}
}
