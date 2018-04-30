package edu.usc.thevillagers.serversideagent.env.sensor;

import edu.usc.thevillagers.serversideagent.agent.Agent;

public class SensorGaussian extends Sensor {

	public SensorGaussian(int dim) {
		super(dim);
	}

	@Override
	public void sense(Agent agent) {
		for(int i = 0; i < dim; i++) 
			values[i] = (float) agent.entity.world.rand.nextGaussian();
	}
}
