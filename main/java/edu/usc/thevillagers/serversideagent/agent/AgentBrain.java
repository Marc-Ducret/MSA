package edu.usc.thevillagers.serversideagent.agent;

public abstract class AgentBrain {
	
	//TODO is it made obsolete by Environment?

	private AgentState state;
	
	public final void init() {
		state = initBrain();
	}
	
	public AgentState getState() {
		return state;
	}
	
	protected abstract AgentState initBrain();
	public abstract void observe() throws Exception;
	public abstract void act() throws Exception;
	public abstract void terminate();
}
