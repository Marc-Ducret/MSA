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
	
	
	protected Agent agent;
	protected AgentBrainEnvironment brain;
	protected WorldServer world;
	
	private Process process;
	private DataOutputStream pOut;
	private DataInputStream pIn;
	
	public Environment(String name, int stateDim, int actionDim, int numberAgents) {
		this.name = name;
		this.stateDim = stateDim;
		this.actionDim = actionDim;
	}
	
	public void init(WorldServer world, String cmd) {
		try {
			this.world = world;
			createAgent();
			startProcess(cmd);
		} catch (Exception e) {
			System.err.println("Could not init environment "+name+" ("+e+")");
		}
	}
	
	private void createAgent() {
		agent = new Agent(world, name);
		agent.spawn(getSpawnPoint());
		brain = new AgentBrainEnvironment(this);
		agent.setBrain(brain);
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
		
		pOut.writeInt(stateDim);
		pOut.writeInt(actionDim);
	}
	
	public final void tick() throws Exception {
		observe();
		step();
		act();
	}
	
	private void observe() throws IOException {
		encodeState(agent, brain.stateVector);
		for(float f : brain.stateVector)
			pOut.writeFloat(f);
	}
	
	private void act() throws IOException {
		for(int i = 0; i < actionDim; i++)
			brain.actionVector[i] = pIn.readFloat();
		decodeAction(agent, brain.actionVector);
	}
	
	protected abstract void encodeState(Agent a, float[] stateVector);
	protected abstract void decodeAction(Agent a, float[] actionVector);
	protected abstract void step() throws Exception;
	
	public abstract void reset();
	
	public abstract BlockPos getSpawnPoint();
}
