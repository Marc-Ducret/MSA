package edu.usc.thevillagers.serversideagent.agent;

public class AgentBrainEnvironment extends AgentBrain {
	
	public final Environment env;
	public final float[] stateVector;
	public final float[] actionVector;
	
	public AgentBrainEnvironment(Environment env) {
		this.env = env;
		stateVector = new float[env.stateDim];
		actionVector = new float[env.actionDim];
	}
	
	@Override
	protected AgentState initBrain() {
		return new AgentState(1, 0);
	}

	@Override
	public void observe() throws Exception {
	}

	@Override
	public void act() throws Exception {
	}

	@Override
	public void terminate() {
	}
}
