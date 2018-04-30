package edu.usc.thevillagers.serversideagent.env;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.env.actuator.ActuatorForwardStrafe;
import edu.usc.thevillagers.serversideagent.env.sensor.SensorPosition;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.WorldServer;

public class EnvironmentPush extends Environment {

	protected BlockPos ref;
	
	private Actor actorA;
	private Actor actorB;
	
	@Override
	public void readPars(float[] pars) {
	}

	@Override
	protected void buildSensors() {
		sensors.add(new SensorPosition(1, 0, 1, (a) -> 
					a.entity.getPositionVector().subtract(new Vec3d(ref))));
		sensors.add(new SensorPosition(1, 0, 1, (a) -> 
					opponent(a).entity.getPositionVector().subtract(new Vec3d(ref))));
	}
	
	@Override
	protected void buildActuators() {
		actuators.add(new ActuatorForwardStrafe());
	}
	
	@Override
	public void init(WorldServer world) {
		super.init(world);
		ref = getOrigin();
	}
	
	@Override
	public void newActor(Actor a) {
		super.newActor(a);
		if(actorA == null) actorA = a;
		else if(actorB == null) {
			actorB = a;
			actorB.envData = actorA;
			actorA.envData = actorB;
		}
	}

	@Override
	protected void stepActor(Actor actor) throws Exception {
		if(actor.entity.posY < ref.getY() - .01F) {
			done = true;
			actor.reward = -10;
			opponent(actor).reward = 10;
		}
	}
	
	private Actor opponent(Actor a) {
		return (Actor) a.envData;
	}
}
