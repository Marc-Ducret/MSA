package edu.usc.thevillagers.serversideagent.env;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.function.Predicate;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.agent.Agent;
import edu.usc.thevillagers.serversideagent.env.actuator.Actuator;
import edu.usc.thevillagers.serversideagent.env.allocation.Allocator;
import edu.usc.thevillagers.serversideagent.env.sensor.Sensor;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;
import net.minecraft.world.WorldServer;

/**
 * Represents an environment for Agents inside of Minecraft.
 */
public abstract class Environment { //TODO document functions that should be overridden
	
	public long emptyTime = 0;
	
	public final String name;
	public String id;
	public final List<Sensor> sensors;
	public final List<Actuator> actuators;
	public int observationDim, actionDim;
	public int entityDim = 0;
	public int entityMax = 0;
	
	protected Allocator allocator;
	private boolean allocated = false;
	
	public WorldServer world;
	
	private BlockPos origin;
	
	private List<Actor> actors = new ArrayList<>();
	
	public boolean done;
	public int time;
	
	public Environment() {
		this.name = getClass().getSimpleName().replaceFirst("Environment", "");
		this.sensors = new ArrayList<>();
		this.actuators = new ArrayList<>();
	}
	
	public abstract void readPars(float[] pars);
	protected abstract void buildSensors();
	protected abstract void buildActuators();
	
	public void encodeEntityObservation(Agent a, List<Float> obs) { //TODO doc
	}
	
	protected int getRoundPar(float[] pars, int i, int def) {
		return Math.round(getPar(pars, i, def));
	}
	
	protected float getPar(float[] pars, int i, float def) {
		if(pars == null || i < 0 || i >= pars.length) return def;
		return pars[i];
	}
	
	public void init(WorldServer world) {
		this.world = world;
		buildSensors();
		buildActuators();
		int observationDim = 0;
		for(Sensor sensor : sensors) observationDim += sensor.dim;
		this.observationDim = observationDim;
		int actionDim = 0;
		for(Actuator actuator : actuators) actionDim += actuator.dim;
		this.actionDim = actionDim;
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
	
	public final void encodeObservation(Actor actor, float[] observationVector) {
		int offset = 0;
		for(Sensor sensor : sensors) {
			sensor.sense(actor);
			for(int i = 0; i < sensor.dim; i ++) 
				observationVector[offset++] = sensor.values[i];
		}
	}
	
	public final void decodeAction(Actor actor, float[] actionVector) {
		int offset = 0;
		for(Actuator actuator : actuators) {
			for(int i = 0; i < actuator.dim; i ++)
				actuator.values[i] = actionVector[offset++];
			actuator.act(actor);
		}
	}
	
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
				if(!(e instanceof IOException)) {
					e.printStackTrace();
				}
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
