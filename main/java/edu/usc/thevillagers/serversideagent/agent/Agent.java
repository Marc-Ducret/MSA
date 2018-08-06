package edu.usc.thevillagers.serversideagent.agent;

import java.io.IOException;

import edu.usc.thevillagers.serversideagent.env.Environment;
import edu.usc.thevillagers.serversideagent.request.DataSocket;
import net.minecraftforge.fml.common.FMLCommonHandler;

/**
 * An autonomous actor. Communicates through a socket for another process to do the thinking.
 */
public class Agent extends Actor {
	
	private DataSocket sok;
	
	public Agent(Environment env, EntityAgent agent, DataSocket sok) throws IOException {
		super(env, agent, agent.actionState);
		this.sok = sok;
	}
	
	@Override
	public void joinEnv(int actorId) throws IOException {
		super.joinEnv(actorId);
		sok.out.writeUTF(env.id);
		sok.out.writeInt(env.observationDim);
		sok.out.writeInt(env.actionDim);
		sok.out.writeInt(env.entityDim);
		sok.out.writeInt(env.entityMax);
		sok.out.writeInt(actorId);
		sok.out.flush();
	}
	
	public void terminate() {
		if(!sok.socket.isClosed()) 
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
		if(env.entityDim > 0) {
			observationEntities.clear();
			env.encodeEntityObservation(this, observationEntities);
			sok.out.writeInt(observationEntities.size());
			for(float f : observationEntities)
				sok.out.writeFloat(f);
		}
	}
	
	public void observe() throws IOException {
		sendObservation();
		sok.out.writeFloat(reward);
		sok.out.writeBoolean(env.done);
		sok.out.flush();
	}
	
	public void observeNoReward() throws IOException {
		sendObservation();
		sok.out.flush();
	}
	
	public void act() throws Exception {
		if(entity.isDead) throw new Exception("is dead");
		FMLCommonHandler.instance().getMinecraftServerInstance().profiler.startSection("waitPython");
		for(int i = 0; i < env.actionDim; i++)
			actionVector[i] = sok.in.readFloat();
		FMLCommonHandler.instance().getMinecraftServerInstance().profiler.endSection();
		env.decodeAction(this, actionVector);
		actionState.clamp();
		sok.socket.setSoTimeout(3000);
	}
}
