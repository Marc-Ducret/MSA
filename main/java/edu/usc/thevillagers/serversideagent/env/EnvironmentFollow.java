package edu.usc.thevillagers.serversideagent.env;

import java.io.IOException;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.env.actuator.ActuatorForwardStrafe;
import edu.usc.thevillagers.serversideagent.env.sensor.SensorPosition;
import net.minecraft.entity.Entity;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.WorldServer;

public class EnvironmentFollow extends Environment {
	
	private Entity target;
	
	@Override
	public void readPars(float[] pars) {
	}

	@Override
	protected void buildSensors() {
		sensors.add(new SensorPosition(5, 0, 5, 
				(a) -> target.getPositionVector().subtract(a.entity.getPositionVector())));
	}
	
	@Override
	protected void buildActuators() {
		actuators.add(new ActuatorForwardStrafe());
	}
	
	@Override
	public void init(WorldServer world) {
		super.init(world);
		BlockPos ref = getOrigin();
		target = world.getClosestPlayer(ref.getX(), ref.getY(), ref.getZ(), 50, false);
	}
	
	@Override
	public void newActor(Actor a) throws IOException {
		super.newActor(a);
		done = true;
	}

	@Override
	protected void stepActor(Actor actor) throws Exception {
		float dx = (float) (target.posX - actor.entity.posX) / 5F;
		float dz = (float) (target.posZ - actor.entity.posZ) / 5F;
		actor.reward = - (dx*dx + dz*dz);
	}
}
