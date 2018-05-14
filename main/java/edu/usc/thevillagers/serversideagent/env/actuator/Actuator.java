package edu.usc.thevillagers.serversideagent.env.actuator;

import edu.usc.thevillagers.serversideagent.agent.Actor;

/**
 * Represents a part of an Environment's action space.
 */
public abstract class Actuator {

	public final int dim;
	public final float[] values;
	
	public Actuator(int dim) {
		this.dim = dim;
		this.values = new float[dim];
	}
	
	public abstract void act(Actor actor);
	public abstract Reverser reverser(Actor actor);
	
	public abstract class Reverser {
		
		public final Actor actor;
		public final float[] values;
		
		public Reverser(Actor actor) {
			this.actor = actor;
			this.values = new float[dim];
		}
		
		public abstract void startStep();
		public abstract void tick();
		public abstract float[] endStep();
	}
}
