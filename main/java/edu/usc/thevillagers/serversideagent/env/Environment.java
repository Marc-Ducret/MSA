package edu.usc.thevillagers.serversideagent.env;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import edu.usc.thevillagers.serversideagent.agent.Agent;
import edu.usc.thevillagers.serversideagent.agent.AgentBrainEnvironment;
import edu.usc.thevillagers.serversideagent.agent.AgentState;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.WorldServer;

public abstract class Environment {
	
	public final String name;
	public final int stateDim, actionDim;
	
	private BlockPos spawnPoint;
	
	protected Agent agent;
	protected AgentBrainEnvironment brain;
	protected WorldServer world;
	
	protected float reward;
	
	private Process process;
	private DataOutputStream pOut;
	private DataInputStream pIn;
	
	
	public Environment(String name, int stateDim, int actionDim) {
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
		pOut.flush();
	}
	
	public BlockPos getSpawnPoint() {
		return spawnPoint;
	}
	
	public void setSpawnPoint(BlockPos spawnPoint) {
		this.spawnPoint = spawnPoint;
	}
	
	public void terminate() {
		if(process != null && process.isAlive()) 
			process.destroy();
	}
	
	public final void preTick() throws Exception {
		act();
	}
	
	public final void postTick() throws Exception {
		step();
		observe();
	}
	
	private void observe() throws IOException {
		encodeState(agent, brain.stateVector);
		for(float f : brain.stateVector)
			pOut.writeFloat(f);
		pOut.writeFloat(reward);
		reward = 0;
		pOut.writeBoolean(false); //TODO implement done
		pOut.flush();
	}
	
	private void act() throws IOException {
		for(int i = 0; i < actionDim; i++)
			brain.actionVector[i] = pIn.readFloat();
		decodeAction(brain.getState(), brain.actionVector);
	}
	
	protected abstract void encodeState(Agent a, float[] stateVector);
	protected abstract void decodeAction(AgentState a, float[] actionVector);
	protected abstract void step() throws Exception;
	
	public abstract void reset();
}
