package edu.usc.thevillagers.serversideagent.env.controller;

import edu.usc.thevillagers.serversideagent.env.Environment;
import edu.usc.thevillagers.serversideagent.request.DataSocket;

public class ControllerPython extends Controller {

	private final DataSocket sok;
	
	public ControllerPython(Environment env, DataSocket sok) {
		super(env);
		this.sok = sok;
	}

	@Override
	public void step(boolean done) {
		try {
			sok.out.writeBoolean(done);
			sok.out.flush();
			state = ControllerState.values()[sok.in.read()];
			if(state == ControllerState.LOAD) stateParam = sok.in.readInt();
		} catch (Exception e) {
			state = ControllerState.TERMINATE;
		}
	}
}
