package edu.usc.thevillagers.serversideagent.env;

import java.util.Arrays;
import java.util.List;

import edu.usc.thevillagers.serversideagent.agent.Actor;
import edu.usc.thevillagers.serversideagent.env.actuator.ActuatorForwardStrafe;
import edu.usc.thevillagers.serversideagent.env.allocation.AllocatorEmptySpace;
import edu.usc.thevillagers.serversideagent.env.sensor.SensorBlocks;
import net.minecraft.block.BlockColored;
import net.minecraft.block.state.IBlockState;
import net.minecraft.init.Blocks;
import net.minecraft.item.EnumDyeColor;
import net.minecraft.util.math.BlockPos;

public class EnvironmentFloorSurvival extends Environment {
	
	private int size;
	private static final List<EnumDyeColor> STAGES = 
			Arrays.asList(new EnumDyeColor[] {EnumDyeColor.GREEN, EnumDyeColor.LIME, EnumDyeColor.YELLOW, EnumDyeColor.ORANGE, EnumDyeColor.RED});
	
	@Override
	public void readPars(float[] pars) {
		size = getRoundPar(pars, 0, 1);
		allocator = new AllocatorEmptySpace(new BlockPos(-size, -1, -size), new BlockPos(size, 2, size));
	}
	
	@Override
	protected void buildSensors() {
		sensors.add(new SensorBlocks(new BlockPos(-size, -1, -size), new BlockPos(size, -1, size), (state) -> {
			if(state.getBlock() == Blocks.STAINED_GLASS) {
				return STAGES.indexOf(state.getValue(BlockColored.COLOR)) / (float) STAGES.size();
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
						Blocks.STAINED_GLASS.getDefaultState().withProperty(BlockColored.COLOR, STAGES.get(0)));
			}
		}
	}

	@Override
	protected void stepActor(Actor a) throws Exception {
		a.reward = 0;
		BlockPos p = a.entity.getPosition().down();
		IBlockState state = world.getBlockState(p);
		if(state.getBlock() == Blocks.STAINED_GLASS) {
			int dmg = STAGES.indexOf(state.getValue(BlockColored.COLOR));
			dmg++;
			if(dmg < STAGES.size()) {
				world.setBlockState(p,
						Blocks.STAINED_GLASS.getDefaultState().withProperty(BlockColored.COLOR, STAGES.get(dmg)));
			} else {
				world.setBlockState(p, Blocks.AIR.getDefaultState());
			}
		} else {
			done = true;
			a.reward = time / 20F;
		}
	}
	
	@Override
	protected void step() {
		super.step();
	}
}
