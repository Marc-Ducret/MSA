package edu.usc.thevillagers.serversideagent.env.controller;

import edu.usc.thevillagers.serversideagent.env.Environment;

/**
 * Default controller, resets after the episode ended and terminates empty allocated environments
 */
public class ControllerDefault extends Controller {

	public ControllerDefault(Environment env) {
		super(env);
	}

	@Override
	public void step(boolean done) {
		state = done ? ControllerState.RESET : ControllerState.WAIT;
		if(done && env.isEmpty() && env.isAllocated()) state = ControllerState.TERMINATE;
	}
}
