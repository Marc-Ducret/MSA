package edu.usc.thevillagers.serversideagent.env;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.function.Predicate;

import edu.usc.thevillagers.serversideagent.agent.Agent;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.WorldServer;

public abstract class Environment {
	
	public final String name;
	public final int observationDim, actionDim;
	
	public WorldServer world;
	
	private BlockPos origin;
	
	private List<Agent> agents = new ArrayList<>();
	
	public boolean done;
	public int time;
	
	public Environment(String name, int stateDim, int actionDim) { //TODO extract name from class name? (to mirror /e behaviour)
		this.name = name;
		this.observationDim = stateDim;
		this.actionDim = actionDim;
	}
	
	public void init(WorldServer world) {
		this.world = world;
	}
	
	public void newAgent(Agent a) {
		agents.add(a);
	}
	
	public BlockPos getOrigin() {
		return origin;
	}
	
	public void setOrigin(BlockPos origin) {
		this.origin = origin.toImmutable();
	}
	
	public void terminate() {
		applyToAllAgents((a) -> a.terminate());
	}
	
	public final void preTick() throws Exception {
		applyToActiveAgents((a) -> a.act());
	}
	
	public final void postTick() throws Exception {
		step();
		time++;
		applyToActiveAgents((a) -> a.observe());
		if(done) {
			applyToInactivAgents((a) -> {
				a.active = true;
			});
			reset();
			applyToActiveAgents((a) -> {
				a.sync(0x13371337);
				a.observeNoReward();
			});
		}
	}
	
	public BlockPos getSpawnPoint(Agent a) {
		return getOrigin();
	}
	
	public abstract void encodeObservation(Agent agent, float[] stateVector);
	public abstract void decodeAction(Agent agent, float[] actionVector);
	protected abstract void stepAgent(Agent a) throws Exception;
	
	protected void step() {
		applyToActiveAgents((a) -> stepAgent(a));
		boolean atLeastOneActive = false;
		for(Agent a : agents) if(a.active) atLeastOneActive = true;
		if(!atLeastOneActive) done = true;
	}
	
	public void reset() {
		done = false;
		time = 0;
		applyToActiveAgents((a) -> resetAgent(a));
	}
	
	public void resetAgent(Agent a) {
		a.entity.moveToBlockPosAndAngles(getSpawnPoint(a), 0, 0);
		a.entity.connection.setPlayerLocation(a.entity.posX, a.entity.posY, a.entity.posZ, a.entity.rotationYaw, a.entity.rotationPitch);
		a.reward = 0;
	}
	
	public void applyToActiveAgents(AgentApplication app) {
		applyToAgents(app, (a) -> a.active);
	}
	
	public void applyToInactivAgents(AgentApplication app) {
		applyToAgents(app, (a) -> !a.active);
	}
	
	public void applyToAllAgents(AgentApplication app) {
		applyToAgents(app, (a) -> true);
	}
	
	private void applyToAgents(AgentApplication app, Predicate<Agent> filter) {
		Iterator<Agent> iter = agents.iterator();
		while(iter.hasNext()) {
			Agent a = iter.next();
			try {
				if(filter.test(a)) app.apply(a);
			} catch(Exception e) {
				a.terminate();
				iter.remove();
				System.out.println("Agent terminated ("+e+")");
			}
		}
	}
	
	private static interface AgentApplication { 
		void apply(Agent a) throws Exception; 
	}
}
