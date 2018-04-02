package edu.usc.thevillagers.serversideagent.agent;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import net.minecraft.util.math.BlockPos;
import net.minecraft.world.WorldServer;

public abstract class Environment {
	
	public final String name;
	public final int stateDim, actionDim;
	
	
	protected final Agent[] agents;
	protected final AgentBrainEnvironment[] brains;
	protected WorldServer world;
	
	private Process process;
	private DataOutputStream pOut;
	private DataInputStream pIn;
	
	public Environment(String name, int stateDim, int actionDim, int numberAgents) {
		this.name = name;
		this.stateDim = stateDim;
		this.actionDim = actionDim;
		this.agents = new Agent[numberAgents];
		this.brains = new AgentBrainEnvironment[numberAgents];
	}
	
	public void init(WorldServer world, String cmd) {
		try {
			this.world = world;
			createAgents();
			startProcess(cmd);
		} catch (Exception e) {
			System.err.println("Could not init environment "+name+" ("+e+")");
		}
	}
	
	private void createAgents() {
		for(int i = 0; i < agents.length; i++) {
			String agentName = name;
			if(agents.length > 1) {
				agentName += " "+(i+1);
			}
			agents[i] = new Agent(world, agentName);
			agents[i].spawn(getSpawnPoint(i));
			brains[i] = new AgentBrainEnvironment(this);
			agents[i].setBrain(brains[i]);
		}
	}
	
	private void startProcess(String cmd) throws IOException {
		process = Runtime.getRuntime().exec(cmd);
		pOut = new DataOutputStream(new BufferedOutputStream(process.getOutputStream()));
		pIn = new DataInputStream(new BufferedInputStream(process.getInputStream()));
		new Thread(() -> {
			BufferedReader err = new BufferedReader(new InputStreamReader(process.getErrorStream()));
			String line;
			try {
				while((line = err.readLine()) != null)
					System.out.println(line);
			} catch (IOException e) {
			}
			System.out.println("Process "+cmd+" terminated");
		}).start();
	}
	
	public final void tick() throws Exception {
		observe();
		step();
		act();
	}
	
	private void observe() throws IOException {
		for(int i = 0; i < agents.length; i++) {
			encodeState(agents[i], brains[i].stateVector);
		}
		for(AgentBrainEnvironment brain : brains)
			for(float f : brain.stateVector)
				pOut.writeFloat(f);
	}
	
	private void act() throws IOException {
		for(AgentBrainEnvironment brain : brains)
			for(int i = 0; i < actionDim; i++)
				brain.actionVector[i] = pIn.readFloat();
		for(int i = 0; i < agents.length; i++) {
			decodeAction(agents[i], brains[i].actionVector);
		}
	}
	
	protected abstract void encodeState(Agent a, float[] stateVector);
	protected abstract void decodeAction(Agent a, float[] actionVector);
	protected abstract void step() throws Exception;
	
	public abstract void reset();
	
	public abstract BlockPos getSpawnPoint(int agent);
}
