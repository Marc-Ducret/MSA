package edu.usc.thevillagers.serversideagent.env;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.env.actuator.ActuatorDummy;
import edu.usc.thevillagers.serversideagent.env.allocation.AllocatorEmptySpace;
import edu.usc.thevillagers.serversideagent.env.sensor.SensorGaussian;
import net.minecraft.block.BlockColored;
import net.minecraft.init.Blocks;
import net.minecraft.item.EnumDyeColor;
import net.minecraft.util.math.BlockPos;

public class EnvironmentIdle extends Environment {
	
	@Override
	public void readPars(float[] pars) {
		allocator = new AllocatorEmptySpace(new BlockPos(-1, -1, -1), new BlockPos(1, 2, 1));
	}
	
	@Override
	protected void buildSensors() {
		sensors.add(new SensorGaussian(2));
	}
	
	@Override
	protected void buildActuators() {
		actuators.add(new ActuatorDummy(1));
	}
	
	@Override
	public void reset() {
		super.reset();
		generate();
	}

	private void generate() {
		world.setBlockState(getOrigin().down(), Blocks.STAINED_GLASS.getDefaultState().withProperty(BlockColored.COLOR, EnumDyeColor.CYAN));
	}

	@Override
	protected void stepActor(Actor a) throws Exception {
		a.reward = 0;
		for(float f : a.actionVector) {
			a.reward -= f*f;
		}
		if(time >= 10)
			done = true;
	}
}
