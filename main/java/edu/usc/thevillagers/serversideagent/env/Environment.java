package edu.usc.thevillagers.serversideagent.env;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.function.Predicate;

import edu.usc.thevillagers.serversideagent.agent.Actor;
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
	
	private List<Actor> actors = new ArrayList<>();
	
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
	
	public void newActor(Actor a) {
		actors.add(a);
	}
	
	public BlockPos getOrigin() {
		return origin;
	}
	
	public void setOrigin(BlockPos origin) {
		this.origin = origin.toImmutable();
	}
	
	public void terminate() {
		applyToAllActors((a) -> a.terminate());
		if(allocator != null) allocator.free(world, origin);
	}
	
	public final void preTick() throws Exception {
		applyToActiveActors((a) -> a.act());
	}
	
	public final void postTick() throws Exception {
		step();
		time++;
		applyToActiveActors((a) -> a.observe());
		if(done) {
			applyToInactivActors((a) -> {
				a.active = true;
			});
			reset();
			applyToActiveActors((a) -> {
				a.observeNoReward();
			});
		}
	}
	
	public BlockPos getSpawnPoint(Actor a) {
		return getOrigin();
	}
	
	public abstract void encodeObservation(Agent agent, float[] stateVector);
	public abstract void decodeAction(Agent agent, float[] actionVector);
	protected abstract void stepActor(Actor a) throws Exception;
	
	protected void step() {
		applyToActiveActors((a) -> stepActor(a));
		boolean atLeastOneActive = false;
		for(Actor a : actors) if(a.active) atLeastOneActive = true;
		if(!atLeastOneActive) done = true;
	}
	
	public void reset() {
		done = false;
		time = 0;
		applyToActiveActors((a) -> resetActor(a));
	}
	
	public void resetActor(Actor a) {
		a.entity.moveToBlockPosAndAngles(getSpawnPoint(a), 0, 0);
		a.entity.connection.setPlayerLocation(a.entity.posX, a.entity.posY, a.entity.posZ, a.entity.rotationYaw, a.entity.rotationPitch);
		a.reward = 0;
	}
	
	public void applyToActiveActors(ActorApplication app) {
		applyToActors(app, (a) -> a.active);
	}
	
	public void applyToInactivActors(ActorApplication app) {
		applyToActors(app, (a) -> !a.active);
	}
	
	public void applyToAllActors(ActorApplication app) {
		applyToActors(app, (a) -> true);
	}
	
	private void applyToActors(ActorApplication app, Predicate<Actor> filter) {
		Iterator<Actor> iter = actors.iterator();
		while(iter.hasNext()) {
			Actor a = iter.next();
			try {
				if(filter.test(a)) app.apply(a);
			} catch(Exception e) {
				a.terminate();
				iter.remove();
				System.out.println("Actor terminated ("+e+")");
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
	
	protected static interface ActorApplication { 
		void apply(Actor a) throws Exception; 
	}
	
	public boolean isAllocated() {
		return allocated;
	}
	
	public boolean isEmpty() {
		return actors.isEmpty();
	}
}
