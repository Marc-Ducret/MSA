package edu.usc.thevillagers.serversideagent.env;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.env.actuator.ActuatorForwardStrafe;
import edu.usc.thevillagers.serversideagent.env.allocation.AllocatorEmptySpace;
import edu.usc.thevillagers.serversideagent.env.sensor.SensorBlocks;
import net.minecraft.block.BlockColored;
import net.minecraft.block.state.IBlockState;
import net.minecraft.init.Blocks;
import net.minecraft.item.EnumDyeColor;
import net.minecraft.util.math.BlockPos;

public class EnvironmentGotoGold extends Environment {
	
	private int size;
	
	@Override
	public void readPars(float[] pars) {
		size = getRoundPar(pars, 0, 5);
		allocator = new AllocatorEmptySpace(new BlockPos(-size, -1, -size), new BlockPos(size, 2, size));
	}
	
	@Override
	protected void buildSensors() {
		sensors.add(new SensorBlocks(new BlockPos(-size, -1, -size), new BlockPos(size, -1, size), (state) -> {
			if(state.getBlock() == Blocks.STAINED_GLASS) {
				return state.getValue(BlockColored.COLOR) == EnumDyeColor.YELLOW ? 1F : 0F;
			} else {
				return 1F;
			}
		}));
	}
	
	@Override
	protected void buildActuators() {
		actuators.add(new ActuatorForwardStrafe());
	}
	
	@Override
	public void reset() {
		super.reset();
		generate();
	}
	
	private void generate() {
		BlockPos ref = getOrigin();
		for(int z =- size; z <= size; z++) {
			for(int x =- size; x <= size; x++) {
				world.setBlockState(ref.add(x, -1, z), 
						Blocks.STAINED_GLASS.getDefaultState().withProperty(BlockColored.COLOR, EnumDyeColor.LIME));
			}
		}
		int x = world.rand.nextInt(2 * size + 1) - size;
		int z = world.rand.nextInt(2 * size + 1) - size;
		world.setBlockState(ref.add(x, -1, z), 
				Blocks.STAINED_GLASS.getDefaultState().withProperty(BlockColored.COLOR, EnumDyeColor.YELLOW));
	}

	@Override
	protected void stepActor(Actor a) throws Exception {
		BlockPos p = a.entity.getPosition().down();
		IBlockState state = world.getBlockState(p);
		a.reward = -.01F;
		if(state.getBlock() == Blocks.STAINED_GLASS) {
			if(state.getValue(BlockColored.COLOR) == EnumDyeColor.YELLOW) {
				done = true;
				a.reward = 10;
			} else if(time > 99) {
				done = true;
				a.reward = -1;
			}
		} else {
			done = true;
			a.reward = -10;
		}
	}
	
	@Override
	protected void step() {
		super.step();
	}
}
