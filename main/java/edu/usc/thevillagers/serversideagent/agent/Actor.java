package edu.usc.thevillagers.serversideagent.agent;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import edu.usc.thevillagers.serversideagent.env.Environment;
import net.minecraft.entity.player.EntityPlayerMP;

/**
 * Abstract class to represent an actor in an environment. An actor can be a human or an agent.
 */
public abstract class Actor {
	
	public final Environment env;
	public final EntityPlayerMP entity;
	public final AgentActionState actionState;
	
	public final float[] observationVector, actionVector;
	public final List<Float> observationEntities;
	
	public float reward;
	
	public boolean active;
	public Object envData;
	
	public Actor(Environment env, EntityPlayerMP entity, AgentActionState actionState) {
		this.entity = entity;
		this.env = env;
		this.observationVector = new float[env.observationDim];
		this.actionVector = new float[env.actionDim];
		this.observationEntities = new ArrayList<>();
		this.active = false;
		this.actionState = actionState;
	}
	
	public void joinEnv(int actorId) throws IOException {}
	
	public abstract void terminate();
	
	public abstract void observe() throws IOException;
	
	public abstract void observeNoReward() throws IOException;
	
	public abstract void act() throws Exception;
}
