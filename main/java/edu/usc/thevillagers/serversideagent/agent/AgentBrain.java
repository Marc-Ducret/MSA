package edu.usc.thevillagers.serversideagent.agent;

public abstract class AgentBrain {

	public final AgentState state;
	
	public AgentBrain(AgentState state) {
		this.state = state;
	}
	
	public abstract void init();
	
	public abstract void observe();
	
	public abstract void act();
	
	public abstract void terminate();
}
