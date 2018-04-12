package edu.usc.thevillagers.serversideagent.agent;

import java.io.IOException;

import edu.usc.thevillagers.serversideagent.env.Environment;
import edu.usc.thevillagers.serversideagent.request.DataSocket;
import net.minecraft.entity.player.EntityPlayerMP;
import net.minecraftforge.fml.common.FMLCommonHandler;

public class Agent { //TODO extract Actor superclass and extend it as Agent and HumanActor?
	
	public final Environment env;
	public final EntityPlayerMP entity;
	public final AgentActionState actionState;
	
	public final float[] observationVector, actionVector;
	
	private DataSocket sok;
	
	public float reward;
	
	public boolean active;
	public Object envData;
	
	
	public Agent(Environment env, EntityPlayerMP human) {
		this(env, human, new AgentActionState());
	}
	
	public Agent(Environment env, EntityAgent agent, DataSocket sok) throws IOException {
		this(env, agent, agent.actionState);
		this.sok = sok;
		sok.out.writeInt(env.observationDim);
		sok.out.writeInt(env.actionDim);
		sok.out.flush();
	}
	
	public Agent(Environment env, EntityPlayerMP entity, AgentActionState actionState) {
		this.entity = entity;
		this.env = env;
		this.observationVector = new float[env.observationDim];
		this.actionVector = new float[env.actionDim];
		this.active = false;
		this.actionState = actionState;
		this.sok = null;
	}
	
	public void terminate() {
		if(sok != null && !sok.socket.isClosed()) 
			try {
				sok.socket.close();
			} catch(Exception e) {}
		if(entity instanceof EntityAgent) 
			((EntityAgent) entity).remove();
	}
	
	private void sendObservation() throws IOException {
		env.encodeObservation(this, observationVector);
		for(float f : observationVector)
			sok.out.writeFloat(f);
	}
	
	public void observe() throws IOException {
		if(sok == null) return;
		sendObservation();
		sok.out.writeFloat(reward);
		sok.out.writeBoolean(env.done);
		sok.out.flush();
	}
	
	public void observeNoReward() throws IOException {
		if(sok == null) return;
		sendObservation();
		sok.out.flush();
	}
	
	public void act() throws Exception {
		if(sok == null) return;
		if(entity.isDead) throw new Exception("is dead");
		FMLCommonHandler.instance().getMinecraftServerInstance().profiler.startSection("waitPython");
		for(int i = 0; i < env.actionDim; i++)
			actionVector[i] = sok.in.readFloat();
		FMLCommonHandler.instance().getMinecraftServerInstance().profiler.endSection();
		env.decodeAction(this, actionVector);
		actionState.clamp();
	}
	
	public void sync(int code) throws Exception {
		if(sok == null) return;
		FMLCommonHandler.instance().getMinecraftServerInstance().profiler.startSection("waitPython");
		if(sok.in.readInt() != code) {
			FMLCommonHandler.instance().getMinecraftServerInstance().profiler.endSection();
			throw new Exception("Expected sync code");
		}
		FMLCommonHandler.instance().getMinecraftServerInstance().profiler.endSection();
	}
}
