package edu.usc.thevillagers.serversideagent.env;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.env.actuator.ActuatorForwardStrafe;
import edu.usc.thevillagers.serversideagent.env.sensor.SensorBlocks;
import edu.usc.thevillagers.serversideagent.env.sensor.SensorPosition;
import net.minecraft.block.BlockColored;
import net.minecraft.block.state.IBlockState;
import net.minecraft.init.Blocks;
import net.minecraft.item.EnumDyeColor;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.WorldServer;

public class EnvironmentParkour extends Environment {

	public int sightDist;
	public int width, length;

	protected BlockPos ref;
	
	@Override
	public void readPars(float[] pars) {
		sightDist = getRoundPar(pars, 0, 3);
		width = getRoundPar(pars, 1, 4);
		length = getRoundPar(pars, 2, 15);
	}
	
	@Override
	public void init(WorldServer world) {
		super.init(world);
		ref = getOrigin();
	}
	
	@Override
	protected void buildSensors() {
		sensors.add(new SensorPosition(width, 0, length, 
				(a) -> a.entity.getPositionVector().subtract(new Vec3d(ref))));
		sensors.add(new SensorBlocks(
				new BlockPos(-sightDist, -1, -sightDist), 
				new BlockPos(+sightDist, -1, +sightDist),
				(b) -> encode(b)));
		
	}
	
	@Override
	protected void buildActuators() {
		actuators.add(new ActuatorForwardStrafe());
	}

	private float encode(IBlockState b) {
		if(b.getBlock() != Blocks.STAINED_GLASS) return -1;
		EnumDyeColor color = b.getValue(BlockColored.COLOR);
		switch(color) {
		case LIME:
			return 10;
		default:
			return 1;
		}
	}
	
	@Override
	protected void stepActor(Actor actor) throws Exception {
		float dz = (float) (actor.entity.posZ - ref.getZ()) / length;
		IBlockState b = world.getBlockState(actor.entity.getPosition().down());
		if(actor.entity.posY < ref.getY() - .01F || time >= 100) {
			actor.reward = 10 * dz;
			done = true;
		} else if(b.getBlock() == Blocks.STAINED_GLASS && b.getValue(BlockColored.COLOR) == EnumDyeColor.LIME) {
			actor.reward = 100;
			done = true;
		} else {
			actor.reward = - .001F;
		}
	}
}
