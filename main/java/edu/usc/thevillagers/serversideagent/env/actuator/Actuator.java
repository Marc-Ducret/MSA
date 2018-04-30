package edu.usc.thevillagers.serversideagent.env.actuator;

import edu.usc.thevillagers.serversideagent.agent.Agent;

public abstract class Actuator {

	public final int dim;
	public final float[] values;
	
	public Actuator(int dim) {
		this.dim = dim;
		this.values = new float[dim];
	}
	
	public abstract void act(Agent agent);
}
