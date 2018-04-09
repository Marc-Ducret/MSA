package edu.usc.thevillagers.serversideagent.agent;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import edu.usc.thevillagers.serversideagent.env.Environment;
import net.minecraft.entity.player.EntityPlayerMP;

public class Agent { //TODO extract Actor superclass and extend it as Agent and HumanActor?
	
	public final Environment env;
	public final EntityPlayerMP entity;
	public final AgentActionState actionState;
	
	public final float[] observationVector, actionVector;
	
	private Process process;
	private DataOutputStream pOut;
	private DataInputStream pIn;
	
	public float reward;
	
	public boolean active;
	public Object envData;
	
	
	public Agent(Environment env, EntityPlayerMP entity) {
		this.entity = entity;
		this.env = env;
		this.observationVector = new float[env.observationDim];
		this.actionVector = new float[env.actionDim];
		this.active = false;
		this.actionState = entity instanceof EntityAgent ? ((EntityAgent)entity).actionState : new AgentActionState();
	}
	
	public void startProcess(String cmd) throws IOException {
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
		
		pOut.writeInt(env.observationDim);
		pOut.writeInt(env.actionDim);
		pOut.flush();
	}
	
	public void terminate() {
		if(process != null && process.isAlive()) 
			process.destroy();
		if(entity instanceof EntityAgent) 
			((EntityAgent) entity).remove();
	}
	
	private void sendObservation() throws IOException {
		env.encodeObservation(this, observationVector);
		for(float f : observationVector)
			pOut.writeFloat(f);
	}
	
	public void observe() throws IOException {
		if(process == null) return;
		sendObservation();
		pOut.writeFloat(reward);
		pOut.writeBoolean(env.done);
		pOut.flush();
	}
	
	public void observeNoReward() throws IOException {
		if(process == null) return;
		sendObservation();
		pOut.flush();
	}
	
	public void act() throws Exception {
		if(process == null) return;
		if(entity.isDead) throw new Exception("is dead");
		for(int i = 0; i < env.actionDim; i++)
			actionVector[i] = pIn.readFloat();
		env.decodeAction(this, actionVector);
	}
	
	public void sync(int code) throws Exception {
		if(process == null) return;
		if(pIn.readInt() != code) {
			throw new Exception("Expected reset code 0x13371337");
		}
	}
	
	public boolean hasAvailableInput() throws IOException {
		if(process == null) return true;
		return pIn.available() > 0;
	}
}
