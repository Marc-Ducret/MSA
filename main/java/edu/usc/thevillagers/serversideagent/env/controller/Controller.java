package edu.usc.thevillagers.serversideagent.env.controller;

import edu.usc.thevillagers.serversideagent.env.Environment;

public abstract class Controller {

	public static enum ControllerState { WAIT, RESET, LOAD, TERMINATE };
	
	protected Environment env;
	public ControllerState state;
	public int stateParam;
	
	public Controller(Environment env) {
		this.env = env;
		this.state = ControllerState.RESET;
	}
	
	public abstract void step(boolean done);
}
