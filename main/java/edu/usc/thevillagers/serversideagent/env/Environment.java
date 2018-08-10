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
import edu.usc.thevillagers.serversideagent.env.controller.Controller;
import edu.usc.thevillagers.serversideagent.env.controller.Controller.ControllerState;
import edu.usc.thevillagers.serversideagent.env.controller.ControllerDefault;
import edu.usc.thevillagers.serversideagent.env.sensor.Sensor;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordReplayer;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;
import net.minecraft.world.WorldServer;
import net.minecraftforge.fml.common.FMLCommonHandler;

/**
 * Represents an environment for Agents inside of Minecraft.
 */
public abstract class Environment {
	
	/**
	 * The name of the environment type. Should be what follows 'Environment' in the class name.
	 */
	public final String name;
	
	/**
	 * Unique name of this specific environment instance.
	 */
	public String id;
	
	/**
	 * Objects describing the observation space.
	 */
	public final List<Sensor> sensors;
	
	/**
	 * Objects describing the action space.
	 */
	public final List<Actuator> actuators;
	
	/**
	 * Sum of sensors' dimensions.
	 */
	public int observationDim;
	
	/**
	 * Sum of actuators' dimensions.
	 */
	public int actionDim;
	
	/**
	 * Dimension of each entity representation (should be 0 if not used).
	 */
	public int entityDim = 0;
	
	/**
	 * Maximum number of entities within an observation (should be 0 if not used).
	 * Only the real number of entities should impact computation time of agents, but they might
	 * allocate memory based on this limit.
	 */
	public int entityMax = 0;
	
	/**
	 * Object describing how the location of the environment can be automatically chosen.
	 * Can be null if allocation should only be done manually.
	 */
	protected Allocator allocator;
	
	/**
	 * Whether this specific environment instance is allocated or not.
	 */
	private boolean allocated = false;
	
	/**
	 * The Minecraft world this environment is in.
	 */
	public WorldServer world;
	
	/**
	 * The location of the environment in the world.
	 */
	private BlockPos origin;
	
	/**
	 * The actors in the environment, some can be inactive if they joined during the current episode.
	 */
	private List<Actor> actors = new ArrayList<>();
	
	/**
	 * The object controlling when the environment resets and loads from replays.
	 */
	private Controller controller;
	
	/**
	 * Whether the terminal state of the episode has been reached.
	 */
	public boolean done;
	
	/**
	 * Number of steps since the beginning of the current episode.
	 */
	public int time;
	
	/**
	 * Should not be called manually.
	 */
	public Environment() {
		this.name = getClass().getSimpleName().replaceFirst("Environment", "");
		this.sensors = new ArrayList<>();
		this.actuators = new ArrayList<>();
		this.controller = new ControllerDefault(this);
	}
	
	/**
	 * Should be treated like a constructor. Any number of parameters should be excepted.
	 * They should be read using {@link #getPar(float[], int, float)} and {@link #getRoundPar(float[], int, int)}.
	 * </br>
	 * Those are specified after the environment's name when making requests as: Name[x,y,z].
	 * @param pars The array of parameters
	 */
	public abstract void initialize(float[] pars);
	
	/**
	 * Should be overridden to populate {@link #sensors}
	 */
	protected abstract void buildSensors();
	
	/**
	 * Should be overridden to populate {@link #actuators}
	 */
	protected abstract void buildActuators();
	
	/**
	 * Encodes the entity-observations. Values should be added to obs. The size of obs should be a multiple
	 * of {@link #entityDim} and at most {@link #entityDim} * {@link #entityMax}.
	 * @param a The observing agent.
	 * @param obs The list to fill.
	 */
	public void encodeEntityObservation(Agent a, List<Float> obs) {
	}
	
<<<<<<< HEAD
	/**
	 * @param pars Parameter array to get from.
	 * @param i Index of the parameter.
	 * @param def Default value to return if the parameter is not specified.
	 * @return round(pars[i]) if possible, otherwise def.
=======
	
	/**
	 * @return Math.round(getPar())
>>>>>>> 94a03b8c7528375faea71710fb79dcd280dd13e8
	 */
	protected int getRoundPar(float[] pars, int i, int def) {
		return Math.round(getPar(pars, i, def));
	}
	
	/**
<<<<<<< HEAD
	 * @param pars Parameter array to get from.
	 * @param i Index of the parameter.
	 * @param def Default value to return if the parameter is not specified.
	 * @return pars[i] if possible, otherwise def.
=======
	 * @param pars: list of par
	 * @param i: index of the par want to get
	 * @param def: default value
	 * @return par[i], or def
>>>>>>> 94a03b8c7528375faea71710fb79dcd280dd13e8
	 */
	protected float getPar(float[] pars, int i, float def) {
		if(pars == null || i < 0 || i >= pars.length) return def;
		return pars[i];
	}
	
	/**
	 * Called when this environment is assigned a World.
	 * @param world The Minecraft world.
	 */
	public void enterWorld(WorldServer world) {
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
	
	/**
	 * Called when an actor join the environment.
	 * @param a The actor
	 */
	public void newActor(Actor a) throws IOException {
		a.joinEnv(actors.size());
		actors.add(a);
	}
	
	/**
	 * Setter for {@link #controller}
	 * @param controller 
	 */
	public void setController(Controller controller) {
		this.controller = controller;
	}
	
	/**
	 * @return {@link #origin}
	 */
	public BlockPos getOrigin() {
		return origin;
	}
	
	/**
	 * Setter for {@link #origin}
	 * @param origin
	 */
	public void setOrigin(BlockPos origin) {
		this.origin = origin.toImmutable();
	}
	
	/**
	 * Called when an environment is removed.
	 */
	public void terminate() {
		applyToAllActors((a) -> a.terminate());
		if(allocator != null) allocator.free(world, origin);
	}
	
	/**
	 * Called at the beginning of ticks
	 */
	public final void preTick() throws Exception {
		if(done) {
			if(controller.state == ControllerState.RESET || controller.state == Controller.ControllerState.LOAD) {
				applyToInactivActors((a) -> {
					a.active = true;
				});
				reset();
				if(controller.state == ControllerState.LOAD) {
					controller.record.seek(controller.stateParam);
					onLoad(controller.record, controller.stateParam);
				}
				applyToActiveActors((a) -> {
					a.observeNoReward();
				});
			}
		}
		if(!done) {
			applyToActiveActors((a) -> a.act());
		}
	}
	
	/**
	 * Called at the end of ticks
	 */
	public final void postTick() throws Exception {
		if(done) {
			controller.step(done);
		} else {
			step();
			time++;
			controller.step(done);
			if(controller.state != ControllerState.WAIT) done = true;
			applyToActiveActors((a) -> a.observe());
		}
		if(controller.state == ControllerState.TERMINATE) throw new Exception("Controller state: TERMINATE");
	}
	
	/**
	 * @param a an Agent
	 * @return Where to spawn the agent
	 */
	public BlockPos getSpawnPoint(Actor a) {
		return getOrigin();
	}
	
	/**
	 * Computes the observation of actor according to {@link #sensors}
	 * @param actor
	 * @param observationVector
	 */
	public final void encodeObservation(Actor actor, float[] observationVector) {
		int offset = 0;
		for(Sensor sensor : sensors) {
			sensor.sense(actor);
			for(int i = 0; i < sensor.dim; i ++) 
				observationVector[offset++] = sensor.values[i];
		}
	}
	
	/**
	 * Reads the action vector of actor according to {@link #actuators}
	 * @param actor
	 * @param actionVector
	 */
	public final void decodeAction(Actor actor, float[] actionVector) {
		actor.actionState.action = null;
		int offset = 0;
		for(Actuator actuator : actuators) {
			for(int i = 0; i < actuator.dim; i ++)
				actuator.values[i] = actionVector[offset++];
			actuator.act(actor);
		}
	}
	
	/**
	 * Called for each active agent at each time step
	 * @param a
	 */
	protected abstract void stepActor(Actor a) throws Exception;
	
	/**
	 * Called at each step
	 */
	protected void step() {
		applyToActiveActors((a) -> stepActor(a));
		boolean atLeastOneActive = false;
		for(Actor a : actors) if(a.active) atLeastOneActive = true;
		if(!atLeastOneActive) done = true;
	}
	
	/**
	 * Called at the start of each episode
	 */
	public void reset() {
		done = false;
		time = 0;
		applyToActiveActors((a) -> resetActor(a));
	}
	
	/**
	 * Called the frame <b>time</b> is loaded from <b>record</b>
	 * @param record
	 * @param time
	 */
	public void onLoad(WorldRecordReplayer record, int time) {
		//TODO set time depending on trajectory progress...
	}
	
	/**
	 * Called at the start of each episode for each agent
	 * @param a
	 */
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
	
	/**
	 * Apply some function to actors verifying some predicate. Exceptions thrown cause the given agent to terminate.
	 * @param app
	 * @param filter
	 */
	private void applyToActors(ActorApplication app, Predicate<Actor> filter) {
		Iterator<Actor> iter = actors.iterator();
		while(iter.hasNext()) {
			Actor a = iter.next();
			try {
				if(filter.test(a)) app.apply(a);
			} catch(Exception e) {
				FMLCommonHandler.instance().getMinecraftServerInstance().addScheduledTask(() -> a.terminate());
				iter.remove();
				System.out.println("Actor terminated ("+e+")");
				if(!(e instanceof IOException)) {
					e.printStackTrace();
				}
			}
		}
	}
	
	/**
	 * Attempts allocation.
	 * @param world
	 * @return Whether it was successful.
	 */
	public boolean tryAllocate(World world) {
		if(allocator == null) return false;
		BlockPos pos = allocator.allocate(world);
		if(pos == null) return false;
		setOrigin(pos);
		allocated = true;
		return true;
	}
	
	/**
	 * Represents a function to be applied to agents.
	 */
	public static interface ActorApplication { 
		void apply(Actor a) throws Exception; 
	}
	
	/**
	 * @return {@link #allocated}
	 */
	public boolean isAllocated() {
		return allocated;
	}
	
	/**
	 * @return Whether there is no agent.
	 */
	public boolean isEmpty() {
		return actors.isEmpty();
	}
}
