package edu.usc.thevillagers.serversideagent.env.actuator;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayer;

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

	@Override
	public Reverser reverser(Actor actor, WorldRecordReplayer replay) {
		return new Reverser(actor) {
			
			@Override
			public void tick() {
			}
			
			@Override
			public void startStep() {
			}
			
			@Override
			public float[] endStep() {
				for(int i = 0; i < dim; i++)
					values[i] = 0;
				return values;
			}
		};
	}
}
