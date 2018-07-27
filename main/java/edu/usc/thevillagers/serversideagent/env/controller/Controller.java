package edu.usc.thevillagers.serversideagent.env.controller;

import java.io.File;
import java.io.IOException;

import edu.usc.thevillagers.serversideagent.env.Environment;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayer;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayerLoad;

public abstract class Controller {

	public static enum ControllerState { WAIT, RESET, LOAD, TERMINATE };
	
	protected Environment env;
	public ControllerState state;
	public int stateParam;
	public WorldRecordReplayer record;
	
	public Controller(Environment env) {
		this.env = env;
		this.state = ControllerState.RESET;
	}
	
	protected void setRecord(String record) throws IOException {
		File recordFile = null;
		for(File file : new File("tmp/records/").listFiles()) {
			if(file.getName().contains(record)) {
				if(recordFile != null) 
					throw new IllegalArgumentException(file.getName()+" and "+recordFile.getName() + " match " + record);
				recordFile = file;
			}
		}
		if(recordFile == null) throw new IllegalArgumentException(record+" not found");
		this.record = new WorldRecordReplayerLoad(recordFile, env);
		this.record.readInfo();
	}
	
	public abstract void step(boolean done);
}
