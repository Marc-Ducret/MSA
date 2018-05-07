package edu.usc.thevillagers.serversideagent.env.sensor;

import edu.usc.thevillagers.serversideagent.agent.Agent;

/**
 * Represents a part of an Environment's constant observation space.
 */
public abstract class Sensor {

	public final int dim;
	public final float[] values;
	
	public Sensor(int dim) {
		this.dim = dim;
		this.values = new float[dim];
	}
	
	public abstract void sense(Agent agent);
}
