package edu.usc.thevillagers.serversideagent.env;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.function.Predicate;

import edu.usc.thevillagers.serversideagent.agent.Agent;
import edu.usc.thevillagers.serversideagent.env.allocation.Allocator;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;
import net.minecraft.world.WorldServer;

public abstract class Environment {
	
	public long emptyTime = 0;
	
	public final String name;
	public String id;
	public final int observationDim, actionDim;
	
	protected Allocator allocator;
	private boolean allocated = false;
	
	public WorldServer world;
	
	private BlockPos origin;
	
	private List<Agent> agents = new ArrayList<>();
	
	public boolean done;
	public int time;
	
	public Environment(int stateDim, int actionDim) {
		this.name = getClass().getSimpleName().replaceFirst("Environment", "");
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
		if(allocator != null) allocator.free(world, origin);
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
//				a.sync(0x13371337);
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
	
	public boolean tryAllocate(World world) {
		if(allocator == null) return false;
		BlockPos pos = allocator.allocate(world);
		if(pos == null) return false;
		setOrigin(pos);
		allocated = true;
		return true;
	}
	
	protected static interface AgentApplication { 
		void apply(Agent a) throws Exception; 
	}
	
	public boolean isAllocated() {
		return allocated;
	}
	
	public boolean isEmpty() {
		return agents.isEmpty();
	}
}
