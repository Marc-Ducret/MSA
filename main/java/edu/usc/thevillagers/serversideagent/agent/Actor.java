package edu.usc.thevillagers.serversideagent.agent;

import java.io.IOException;

import edu.usc.thevillagers.serversideagent.env.Environment;
import net.minecraft.entity.player.EntityPlayerMP;

public abstract class Actor {
	
	public final Environment env;
	public final EntityPlayerMP entity;
	public final AgentActionState actionState;
	
	public final float[] observationVector, actionVector;
	
	public float reward;
	
	public boolean active;
	public Object envData;
	
	public Actor(Environment env, EntityPlayerMP entity, AgentActionState actionState) {
		this.entity = entity;
		this.env = env;
		this.observationVector = new float[env.observationDim];
		this.actionVector = new float[env.actionDim];
		this.active = false;
		this.actionState = actionState;
	}
	
	public abstract void terminate();
	
	public abstract void observe() throws IOException;
	
	public abstract void observeNoReward() throws IOException;
	
	public abstract void act() throws Exception;
}
