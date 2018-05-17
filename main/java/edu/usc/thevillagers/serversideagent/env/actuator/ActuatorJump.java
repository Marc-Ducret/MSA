package edu.usc.thevillagers.serversideagent.env.actuator;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayer;

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

	@Override
	public Reverser reverser(Actor actor, WorldRecordReplayer replay) {
		// TODO Auto-generated method stub
		return null;
	}
}
