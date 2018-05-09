package edu.usc.thevillagers.serversideagent.env.sensor;

import edu.usc.thevillagers.serversideagent.agent.Actor;

/**
 * A sensor that takes random gaussian values.
 */
public class SensorGaussian extends Sensor {

	public SensorGaussian(int dim) {
		super(dim);
	}

	@Override
	public void sense(Actor actor) {
		for(int i = 0; i < dim; i++) 
			values[i] = (float) actor.entity.world.rand.nextGaussian();
	}
}
